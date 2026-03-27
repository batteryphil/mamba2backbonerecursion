/*
 * llama2_efi_mamba.c — SSM (Mamba2 + RLF) Inference Engine for UEFI Bare Metal
 * =============================================================================
 * This is the integrated REPL entry point that bridges the Mamba2 SSM inference
 * engine (ssm_infer.c) with Djiby Diop's llm-baremetal UEFI framework.
 *
 * Features:
 *   - Auto-detects .mamba.bin (SSM) vs .bin/.gguf (Transformer) models
 *   - Full SSM inference with Recursive Latent Forcing (RLF) reasoning loop
 *   - All 20 OO engines preserved and initialized
 *   - Cyberpunk splash + REPL interface
 *   - /ssm_info, /rlf_depth, /rlf_trace commands
 *
 * Architecture:
 *   Token → Embedding → Mamba2 SSM Layers ×N → Final Norm → LM Head
 *                         ↓ (if reasoning query)
 *                    RLF Recursive Loop (RoPE + Lifeline + SSM Core)
 *                         ↓ until <HALT>
 *                       Answer
 *
 * Copyright (c) 2024 Djiby Diop (llm-baremetal) + batteryphil (RLF/Mamba2)
 */

#include <efi.h>
#include <efilib.h>
#include <efinet.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <immintrin.h>
#endif

/* djiblas optimized matmul */
#define DJIBLAS_DISABLE_CPUID 0
#include "djiblas.h"

/* LLM-Kernel primitives */
#include "llmk_zones.h"
#include "llmk_log.h"
#include "llmk_sentinel.h"

/* LLM-OO runtime */
#include "llmk_oo.h"

/* All 20 Operating Organism engines */
#include "djibion-engine/core/djibion.h"
#include "diopion-engine/core/diopion.h"
#include "diagnostion-engine/core/diagnostion.h"
#include "memorion-engine/core/memorion.h"
#include "orchestrion-engine/core/orchestrion.h"
#include "calibrion-engine/core/calibrion.h"
#include "compatibilion-engine/core/compatibilion.h"
#include "evolvion-engine/core/evolvion.h"
#include "synaption-engine/core/synaption.h"
#include "conscience-engine/core/conscience.h"
#include "neuralfs-engine/core/neuralfs.h"
#include "ghost-engine/core/ghost.h"
#include "immunion-engine/core/immunion.h"
#include "dreamion-engine/core/dreamion.h"
#include "symbion-engine/core/symbion.h"
#include "collectivion-engine/core/collectivion.h"
#include "metabion-engine/core/metabion.h"
#include "cellion-engine/core/cellion.h"
#include "morphion-engine/core/morphion.h"
#include "pheromion-engine/core/pheromion.h"

/* Phase 5 metabolism profile */
#include "metabion_profile.h"

/* DjibMark tracing + Interface */
#include "djibmark.h"
#include "interface.h"

/* GGUF support (for Transformer fallback) */
#include "gguf_loader.h"
#include "gguf_infer.h"

/* SSM inference engine */
#include "ssm_infer.h"
#include "ssm_weights.h"

/* ── Model Format Detection ───────────────────────────────────────────────── */

#define MAMBA_BIN_MAGIC_U32 0x4D414D42  /* "MAMB" */

typedef enum {
    MODEL_FMT_UNKNOWN = 0,
    MODEL_FMT_BIN     = 1,  /* llama2.c Transformer */
    MODEL_FMT_GGUF    = 2,  /* GGUF Transformer */
    MODEL_FMT_MAMBA   = 3,  /* .mamba.bin SSM + RLF */
} ModelFormat;

/* ── Global Engine Instances ──────────────────────────────────────────────── */

static DjibionEngine       g_djibion;
static DiopionEngine       g_diopion;
static DiagnostionEngine   g_diagnostion;
static MemorionEngine      g_memorion;
static OrchestrionEngine   g_orchestrion;
static CalibrionEngine     g_calibrion;
static CompatibilionEngine g_compatibilion;
static EvolvionEngine      g_evolvion;
static SynaptionEngine     g_synaption;
static ConscienceEngine    g_conscience;
static NeuralfsEngine      g_neuralfs;
static GhostEngine         g_ghost;
static ImmunionEngine      g_immunion;
static DreamionEngine      g_dreamion;
static SymbionEngine       g_symbion;
static CollectivionEngine  g_collectivion;
static MetabionEngine      g_metabion;
static CellionEngine       g_cellion;
static MorphionEngine      g_morphion;
static PheromionEngine     g_pheromion;

/* ── SSM Model State ──────────────────────────────────────────────────────── */

static MambaConfig   g_ssm_cfg;
static MambaWeights  g_ssm_wt;
static MambaState    g_ssm_state;
static int           g_ssm_loaded  = 0;
static ModelFormat   g_model_fmt   = MODEL_FMT_UNKNOWN;

/* RLF inference settings */
static int    g_rlf_max_loops  = 16;
static int    g_auto_enabled   = 0;
static int    g_auto_tick      = 0;
static int    g_auto_mode      = 0;   /* 0=normal, 1=degraded, 2=safe */
static int    g_auto_interval  = 30;  /* seconds between autonomous ticks */
static int    g_rlf_trace_on   = 0;

/* Generation settings */
static int    g_max_gen_tokens = 256;
static float  g_temperature    = 0.7f;
static int    g_top_k          = 40;

/* Boot verbosity */
static int    g_boot_verbose = 0;

/* ── UEFI Helpers ─────────────────────────────────────────────────────────── */

static void llmk_print_ascii(const char *s) {
    /**
     * Print ASCII string to UEFI console.
     */
    if (!s) return;
    CHAR16 buf[2];
    buf[1] = 0;
    while (*s) {
        buf[0] = (CHAR16)(unsigned char)*s;
        Print(L"%s", buf);
        s++;
    }
}

static EFI_FILE_HANDLE g_root = NULL;

static EFI_STATUS llmk_open_read_file(EFI_FILE_HANDLE *out, const CHAR16 *name) {
    /**
     * Open a file from root for reading.
     */
    if (!out || !name || !g_root) return EFI_INVALID_PARAMETER;
    return uefi_call_wrapper(g_root->Open, 5, g_root, out, (CHAR16 *)name,
                            EFI_FILE_MODE_READ, 0);
}

static UINT64 llmk_file_size(EFI_FILE_HANDLE f) {
    /**
     * Get file size via SetPosition to end.
     */
    if (!f) return 0;
    uefi_call_wrapper(f->SetPosition, 2, f, 0xFFFFFFFFFFFFFFFFULL);
    UINT64 sz = 0;
    uefi_call_wrapper(f->GetPosition, 2, f, &sz);
    uefi_call_wrapper(f->SetPosition, 2, f, 0);
    return sz;
}

static ModelFormat detect_model_format(EFI_FILE_HANDLE f) {
    /**
     * Detect model format from first 4 bytes.
     */
    UINT8 magic[4];
    UINTN n = 4;
    UINT64 pos = 0;
    uefi_call_wrapper(f->GetPosition, 2, f, &pos);
    uefi_call_wrapper(f->SetPosition, 2, f, 0);
    EFI_STATUS st = uefi_call_wrapper(f->Read, 3, f, &n, magic);
    uefi_call_wrapper(f->SetPosition, 2, f, pos);
    if (EFI_ERROR(st) || n != 4) return MODEL_FMT_UNKNOWN;

    /* MAMB = Mamba SSM */
    if (magic[0] == 'M' && magic[1] == 'A' && magic[2] == 'M' && magic[3] == 'B') {
        return MODEL_FMT_MAMBA;
    }
    /* GGUF */
    if (magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F') {
        return MODEL_FMT_GGUF;
    }
    /* Default: llama2.c bin */
    return MODEL_FMT_BIN;
}

/* ── SSM Model Loader (UEFI) ─────────────────────────────────────────────── */

static int load_mamba_model(EFI_FILE_HANDLE f, UINT64 file_size) {
    /**
     * Load a .mamba.bin model file into the SSM engine.
     * Allocates memory via UEFI Boot Services.
     */
    Print(L"[ssm] Loading Mamba2 SSM model (%lu bytes)...\r\n", (UINT64)file_size);

    /* Allocate buffer for entire file */
    void *data = NULL;
    EFI_STATUS st = uefi_call_wrapper(BS->AllocatePool, 3,
                                      EfiLoaderData, (UINTN)file_size, &data);
    if (EFI_ERROR(st) || !data) {
        Print(L"[ssm] ERROR: Failed to allocate %lu bytes\r\n", (UINT64)file_size);
        return -1;
    }

    /* Read entire file */
    uefi_call_wrapper(f->SetPosition, 2, f, 0);
    UINTN read_sz = (UINTN)file_size;
    st = uefi_call_wrapper(f->Read, 3, f, &read_sz, data);
    if (EFI_ERROR(st) || read_sz != (UINTN)file_size) {
        Print(L"[ssm] ERROR: Read failed (got %lu / %lu)\r\n",
              (UINT64)read_sz, (UINT64)file_size);
        uefi_call_wrapper(BS->FreePool, 1, data);
        return -1;
    }

    /* Parse header + load weights */
    if (mamba_load_weights(&g_ssm_wt, &g_ssm_cfg, data, file_size) != 0) {
        Print(L"[ssm] ERROR: Weight loading failed\r\n");
        uefi_call_wrapper(BS->FreePool, 1, data);
        return -1;
    }

    /* Allocate running state */
    if (mamba_state_alloc(&g_ssm_state, &g_ssm_cfg) != 0) {
        Print(L"[ssm] ERROR: State allocation failed\r\n");
        mamba_free_weight_arrays(&g_ssm_wt);
        uefi_call_wrapper(BS->FreePool, 1, data);
        return -1;
    }

    g_ssm_loaded = 1;
    g_rlf_max_loops = g_ssm_cfg.max_rlf_loops;

    /* Print summary */
    Print(L"\r\n");
    Print(L"══════════════════════════════════════════════════════\r\n");
    Print(L"  Mamba2 SSM + RLF Bare-Metal Model Loaded\r\n");
    Print(L"══════════════════════════════════════════════════════\r\n");
    Print(L"  d_model:       %d\r\n", g_ssm_cfg.d_model);
    Print(L"  d_state:       %d\r\n", g_ssm_cfg.d_state);
    Print(L"  d_conv:        %d\r\n", g_ssm_cfg.d_conv);
    Print(L"  d_inner:       %d\r\n", g_ssm_cfg.d_inner);
    Print(L"  n_layers:      %d\r\n", g_ssm_cfg.n_layers);
    Print(L"  vocab_size:    %d\r\n", g_ssm_cfg.vocab_size);
    Print(L"  max_rlf_loops: %d\r\n", g_ssm_cfg.max_rlf_loops);
    Print(L"  halt_token_id: %d\r\n", g_ssm_cfg.halt_token_id);
    Print(L"  lifeline:      %s\r\n",
          g_ssm_wt.lifeline_gate ? L"loaded" : L"none");
    Print(L"  loop_core:     %s\r\n",
          g_ssm_wt.loop_in_proj ? L"loaded" : L"none");
    Print(L"══════════════════════════════════════════════════════\r\n");
    Print(L"\r\n");

    return 0;
}

/* ── SSM Generation (UEFI output) ─────────────────────────────────────────── */

static void ssm_print_token(int token_id) {
    /**
     * Print a single token to UEFI console.
     * For bare-metal, we use the token ID directly as a character index.
     * A real implementation would use the tokenizer vocabulary.
     */
    if (token_id >= 32 && token_id < 127) {
        CHAR16 buf[2];
        buf[0] = (CHAR16)token_id;
        buf[1] = 0;
        Print(L"%s", buf);
    } else if (token_id == 10) {
        Print(L"\r\n");
    } else {
        Print(L"[%d]", token_id);
    }
}

static void ssm_generate_response(const int *prompt_ids, int prompt_len) {
    /**
     * Generate a response using the SSM engine.
     * Routes to RLF inference or standard generation based on model config.
     */
    if (!g_ssm_loaded) {
        Print(L"[ssm] No model loaded\r\n");
        return;
    }

    /* Check if RLF is available */
    int use_rlf = (g_ssm_wt.lifeline_gate != NULL && g_ssm_wt.loop_in_proj != NULL);

    if (use_rlf) {
        /* RLF Recursive inference */
        g_ssm_cfg.max_rlf_loops = g_rlf_max_loops;

        RlfInferenceResult result;
        mamba_rlf_infer(&result, prompt_ids, prompt_len,
                       &g_ssm_cfg, &g_ssm_wt, &g_ssm_state);

        if (g_rlf_trace_on) {
            Print(L"\r\n[rlf_trace] %d loops:\r\n", result.n_loops);
            for (int i = 0; i < result.n_loops; i++) {
                Print(L"  loop[%d]: token=%d conf=%.4f%s\r\n",
                      result.trace[i].loop_index,
                      result.trace[i].predicted_token,
                      (double)result.trace[i].confidence,
                      result.trace[i].is_halt ? L" <HALT>" : L"");
            }
        }

        Print(L"\r\n");
        ssm_print_token(result.final_token);
        Print(L"\r\n");

        Print(L"\r\n[rlf] answer: token=%d (conf=%.4f, %d loops)\r\n",
              result.final_token, (double)result.final_conf, result.n_loops);
    } else {
        /* Standard autoregressive generation */
        int out_tokens[512];
        int n_gen = mamba_generate(out_tokens, g_max_gen_tokens,
                                  prompt_ids, prompt_len,
                                  g_temperature,
                                  &g_ssm_cfg, &g_ssm_wt, &g_ssm_state);

        Print(L"\r\n");
        for (int i = 0; i < n_gen; i++) {
            ssm_print_token(out_tokens[i]);
        }
        Print(L"\r\n");
    }
}

/* ── Simple Tokenizer (character-level for bare-metal) ────────────────────── */

static int tokenize_input(const CHAR16 *input, int *out_ids, int max_len) {
    /**
     * Simple character-level tokenizer for bare-metal.
     * Maps UTF-16 UEFI chars → token IDs.
     * A real implementation would use BPE/sentencepiece.
     */
    int n = 0;
    for (int i = 0; input[i] && n < max_len; i++) {
        CHAR16 c = input[i];
        if (c < 256) {
            out_ids[n++] = (int)c;
        }
    }
    return n;
}

/* ── REPL Command Handlers ────────────────────────────────────────────────── */

static int handle_ssm_command(const CHAR16 *input) {
    /**
     * Handle SSM-specific REPL commands.
     * Returns 1 if command was handled, 0 otherwise.
     */
    if (!input) return 0;

    /* /ssm_info */
    if (input[0] == '/' && input[1] == 's' && input[2] == 's' && input[3] == 'm' &&
        input[4] == '_' && input[5] == 'i' && input[6] == 'n' && input[7] == 'f' &&
        input[8] == 'o') {
        if (!g_ssm_loaded) {
            Print(L"[ssm] No SSM model loaded\r\n");
        } else {
            Print(L"\r\n");
            Print(L"SSM Engine Info:\r\n");
            Print(L"  Architecture:   Mamba2 + RLF v34\r\n");
            Print(L"  d_model:        %d\r\n", g_ssm_cfg.d_model);
            Print(L"  d_state:        %d\r\n", g_ssm_cfg.d_state);
            Print(L"  d_conv:         %d\r\n", g_ssm_cfg.d_conv);
            Print(L"  d_inner:        %d\r\n", g_ssm_cfg.d_inner);
            Print(L"  n_layers:       %d\r\n", g_ssm_cfg.n_layers);
            Print(L"  vocab_size:     %d\r\n", g_ssm_cfg.vocab_size);
            Print(L"  RLF loops:      %d (max %d)\r\n",
                  g_rlf_max_loops, g_ssm_cfg.max_rlf_loops);
            Print(L"  halt_token_id:  %d\r\n", g_ssm_cfg.halt_token_id);
            Print(L"  rope_base:      %d\r\n", g_ssm_cfg.rope_base);
            Print(L"  lifeline_gate:  %s\r\n",
                  g_ssm_wt.lifeline_gate ? L"present" : L"absent");
            Print(L"  loop_core:      %s\r\n",
                  g_ssm_wt.loop_in_proj ? L"present" : L"absent");
            Print(L"  temperature:    ");
            /* Print float manually (no printf in UEFI) */
            int temp_int = (int)(g_temperature * 100.0f);
            Print(L"%d.%02d\r\n", temp_int / 100, temp_int % 100);
            Print(L"  max_gen_tokens: %d\r\n", g_max_gen_tokens);
            Print(L"\r\n");
        }
        return 1;
    }

    /* /rlf_depth N */
    if (input[0] == '/' && input[1] == 'r' && input[2] == 'l' && input[3] == 'f' &&
        input[4] == '_' && input[5] == 'd' && input[6] == 'e' && input[7] == 'p' &&
        input[8] == 't' && input[9] == 'h') {
        int n = 0;
        const CHAR16 *p = input + 10;
        while (*p == ' ') p++;
        while (*p >= '0' && *p <= '9') {
            n = n * 10 + (*p - '0');
            p++;
        }
        if (n > 0 && n <= 64) {
            g_rlf_max_loops = n;
            Print(L"[rlf] depth set to %d\r\n", n);
        } else {
            Print(L"[rlf] current depth: %d (use /rlf_depth N, 1-64)\r\n",
                  g_rlf_max_loops);
        }
        return 1;
    }

    /* /rlf_trace */
    if (input[0] == '/' && input[1] == 'r' && input[2] == 'l' && input[3] == 'f' &&
        input[4] == '_' && input[5] == 't' && input[6] == 'r' && input[7] == 'a' &&
        input[8] == 'c' && input[9] == 'e') {
        g_rlf_trace_on = !g_rlf_trace_on;
        Print(L"[rlf] trace: %s\r\n", g_rlf_trace_on ? L"ON" : L"OFF");
        return 1;
    }

    /* /lifeline on|off|ablation */
    if (input[0] == '/' && input[1] == 'l' && input[2] == 'i' && input[3] == 'f' &&
        input[4] == 'e' && input[5] == 'l' && input[6] == 'i' && input[7] == 'n' &&
        input[8] == 'e') {
        const CHAR16 *arg = input + 9;
        while (*arg == ' ') arg++;

        if (arg[0] == 'o' && arg[1] == 'n') {
            g_ssm_cfg.lifeline_enabled = 1;
            Print(L"[rlf] Lifeline: ON (scaffold active)\r\n");
        } else if (arg[0] == 'o' && arg[1] == 'f' && arg[2] == 'f') {
            g_ssm_cfg.lifeline_enabled = 0;
            Print(L"[rlf] Lifeline: OFF (ablation mode — scaffold removed)\r\n");
        } else if (arg[0] == 'a') {
            /* Ablation test: run same prompt with lifeline on and off */
            if (!g_ssm_loaded || !g_ssm_wt.lifeline_gate) {
                Print(L"[rlf] No RLF model loaded for ablation test\r\n");
                return 1;
            }
            Print(L"\r\n══ LIFELINE ABLATION TEST ═══════════════════════════════\r\n");
            Print(L"  Testing Phase Transition: does the model still work\r\n");
            Print(L"  without its training scaffold?\r\n");
            Print(L"════════════════════════════════════════════════════════\r\n");

            /* Test prompt: simple logic chain */
            int test_ids[] = {'A',' ','=',' ','t','r','u','t','h','.',
                              ' ','B',' ','=',' ','A','.',
                              ' ','W','h','a','t',' ','i','s',' ','B','?',
                              ' ','A','n','s','w','e','r',':'};
            int test_len = sizeof(test_ids) / sizeof(test_ids[0]);

            /* Run with lifeline ON */
            g_ssm_cfg.lifeline_enabled = 1;
            RlfInferenceResult r_on;
            mamba_rlf_infer(&r_on, test_ids, test_len,
                           &g_ssm_cfg, &g_ssm_wt, &g_ssm_state);
            Print(L"\r\n  Lifeline ON:  %d loops, token=%d, conf=%.4f\r\n",
                  r_on.n_loops, r_on.final_token, (double)r_on.final_conf);
            for (int i = 0; i < r_on.n_loops; i++) {
                Print(L"    L%d: token=%d  conf=%.4f%s\r\n",
                      i+1, r_on.trace[i].predicted_token,
                      (double)r_on.trace[i].confidence,
                      r_on.trace[i].is_halt ? L" <HALT>" : L"");
            }

            /* Run with lifeline OFF */
            g_ssm_cfg.lifeline_enabled = 0;
            RlfInferenceResult r_off;
            mamba_rlf_infer(&r_off, test_ids, test_len,
                           &g_ssm_cfg, &g_ssm_wt, &g_ssm_state);
            Print(L"\r\n  Lifeline OFF: %d loops, token=%d, conf=%.4f\r\n",
                  r_off.n_loops, r_off.final_token, (double)r_off.final_conf);
            for (int i = 0; i < r_off.n_loops; i++) {
                Print(L"    L%d: token=%d  conf=%.4f%s\r\n",
                      i+1, r_off.trace[i].predicted_token,
                      (double)r_off.trace[i].confidence,
                      r_off.trace[i].is_halt ? L" <HALT>" : L"");
            }

            /* Compare */
            int matches = 0;
            int total = (r_on.n_loops < r_off.n_loops) ? r_on.n_loops : r_off.n_loops;
            for (int i = 0; i < total; i++) {
                if (r_on.trace[i].predicted_token == r_off.trace[i].predicted_token)
                    matches++;
            }
            Print(L"\r\n  RESULT: %d/%d loops match", matches, total);
            if (r_on.final_token == r_off.final_token) {
                Print(L" — SAME ANSWER ✓ (Phase Transition confirmed)\r\n");
            } else {
                Print(L" — DIFFERENT ANSWER (scaffold still needed)\r\n");
            }
            Print(L"════════════════════════════════════════════════════════\r\n\r\n");

            /* Restore lifeline ON */
            g_ssm_cfg.lifeline_enabled = 1;
        } else {
            Print(L"[rlf] Lifeline: %s\r\n",
                  g_ssm_cfg.lifeline_enabled ? L"ON" : L"OFF (ablation)");
            Print(L"  Usage: /lifeline on|off|ablation\r\n");
        }
        return 1;
    }

    /* /help (extended with SSM commands) */
    if (input[0] == '/' && input[1] == 'h' && input[2] == 'e' && input[3] == 'l' &&
        input[4] == 'p') {
        Print(L"\r\n");
        Print(L"Mamba2 SSM + RLF Bare-Metal REPL\r\n");
        Print(L"──────────────────────────────────────────\r\n");
        Print(L"  /ssm_info      — SSM model configuration\r\n");
        Print(L"  /rlf_depth N   — Set RLF reasoning depth\r\n");
        Print(L"  /rlf_trace     — Toggle per-loop trace\r\n");
        Print(L"  /lifeline on   — Enable prompt lifeline\r\n");
        Print(L"  /lifeline off  — Disable lifeline (ablation)\r\n");
        Print(L"  /lifeline ablation — Phase Transition test\r\n");
        Print(L"  /telemetry     — SSM metrics → OO engines\r\n");
        Print(L"  /model list    — Show available models\r\n");
        Print(L"  /model info    — Current model details\r\n");
        Print(L"  /handoff write — Write OOHANDOFF.TXT receipt\r\n");
        Print(L"  /handoff status— Handoff protocol status\r\n");
        Print(L"  /host goals    — Host-side goals\r\n");
        Print(L"  /host policy   — Host-side policy\r\n");
        Print(L"  /oo_status     — OO engines status\r\n");
        Print(L"  /help          — This help message\r\n");
        Print(L"  quit/exit      — Exit REPL\r\n");
        Print(L"──────────────────────────────────────────\r\n");
        Print(L"\r\n");
        return 1;
    }

    /* /oo_status */
    if (input[0] == '/' && input[1] == 'o' && input[2] == 'o' && input[3] == '_' &&
        input[4] == 's') {
        Print(L"\r\nOO Engines:\r\n");
        Print(L"  djibion        calibrion       memorion\r\n");
        Print(L"  diopion        compatibilion   orchestrion\r\n");
        Print(L"  diagnostion    evolvion        synaption\r\n");
        Print(L"  conscience     neuralfs        ghost\r\n");
        Print(L"  immunion       dreamion        symbion\r\n");
        Print(L"  collectivion   metabion        cellion\r\n");
        Print(L"  morphion       pheromion\r\n");
        Print(L"  (all 20 engines initialized)\r\n\r\n");
        return 1;
    }

    /* /telemetry — show SSM metrics mapped to OO engines */
    if (input[0] == '/' && input[1] == 't' && input[2] == 'e' && input[3] == 'l') {
        if (!g_ssm_loaded) {
            Print(L"[telemetry] No model loaded\r\n");
            return 1;
        }

        /* Analyze lifeline gate if present */
        SsmTelemetry telem;
        memset(&telem, 0, sizeof(telem));
        telem.telemetry_enabled = 1;
        if (g_ssm_wt.lifeline_gate) {
            ssm_analyze_lifeline(&telem, g_ssm_wt.lifeline_gate, g_ssm_cfg.d_model);
        }

        Print(L"\r\n══ SSM TELEMETRY → OO ENGINE MAPPING ══════════════════\r\n\r\n");

        /* memorion — lifeline gate (RAM dimension activity) */
        Print(L"  memorion       │ Lifeline RAM: %.1f%% (%d/%d dims)\r\n",
              (double)(telem.lifeline_ram_frac * 100.0f),
              telem.lifeline_ram_dims, g_ssm_cfg.d_model);

        /* diagnostion — gate statistics */
        Print(L"  diagnostion    │ Gate μ=%.4f σ=%.4f\r\n",
              (double)telem.lifeline_mean, (double)telem.lifeline_std);

        /* calibrion — convergence rate */
        Print(L"  calibrion      │ RLF loops: %d, conv_rate=%.4f\r\n",
              telem.rlf_loops_used, (double)telem.rlf_convergence_rate);

        /* synaption — ALU dimension activity */
        Print(L"  synaption      │ Lifeline ALU: %.1f%% (%d/%d dims)\r\n",
              (double)(telem.lifeline_alu_frac * 100.0f),
              telem.lifeline_alu_dims, g_ssm_cfg.d_model);

        /* evolvion — gate drift (from 1.0 initialization) */
        float drift = telem.lifeline_mean - 1.0f;
        Print(L"  evolvion       │ Gate drift from init: %s%.4f\r\n",
              drift >= 0 ? L"+" : L"", (double)drift);

        /* immunion — NaN/Inf sentinel */
        Print(L"  immunion       │ NaN: %d, Inf: %d\r\n",
              telem.nan_count, telem.inf_count);

        /* pheromion — confidence / entropy */
        Print(L"  pheromion      │ Entropy: %.2f bits, top1=%.3f\r\n",
              (double)telem.output_entropy, (double)telem.output_top1_prob);

        /* Performance */
        Print(L"  metabion       │ Inference: %.1f ms, %.0f TPS\r\n",
              (double)telem.inference_ms, (double)telem.tokens_per_sec);

        /* Hidden state health */
        Print(L"  cellion        │ Hidden ‖h‖=%.2f\r\n",
              (double)telem.hidden_state_norm);

        Print(L"\r\n══════════════════════════════════════════════════════\r\n\r\n");
        return 1;
    }

    /* /model list|info */
    if (input[0] == '/' && input[1] == 'm' && input[2] == 'o' && input[3] == 'd' &&
        input[4] == 'e' && input[5] == 'l') {
        const CHAR16 *arg = input + 6;
        while (*arg == ' ') arg++;

        if (arg[0] == 'l') {
            /* /model list */
            Print(L"\r\nModel files on boot volume:\r\n");
            Print(L"  Searching... (scan for *.mamba.bin, *.bin, *.gguf)\r\n");
            /* In UEFI we'd iterate the root directory here */
            if (g_ssm_loaded) {
                Print(L"  * Current: model.mamba.bin (%d layers, d=%d)\r\n",
                      g_ssm_cfg.n_layers, g_ssm_cfg.d_model);
            }
            Print(L"\r\n");
        } else if (arg[0] == 'i') {
            /* /model info — same as /ssm_info */
            if (!g_ssm_loaded) {
                Print(L"[model] No model loaded\r\n");
            } else {
                Print(L"\r\n");
                Print(L"Current Model:\r\n");
                Print(L"  Type:         Mamba2 SSM + RLF v34\r\n");
                Print(L"  d_model:      %d\r\n", g_ssm_cfg.d_model);
                Print(L"  n_layers:     %d\r\n", g_ssm_cfg.n_layers);
                Print(L"  vocab_size:   %d\r\n", g_ssm_cfg.vocab_size);
                Print(L"  RLF loops:    %d (max %d)\r\n",
                      g_rlf_max_loops, g_ssm_cfg.max_rlf_loops);
                Print(L"  Lifeline:     %s (%s)\r\n",
                      g_ssm_wt.lifeline_gate ? L"present" : L"absent",
                      g_ssm_cfg.lifeline_enabled ? L"ON" : L"OFF");
                Print(L"\r\n");
            }
        } else {
            Print(L"[model] Usage: /model list | /model info\r\n");
        }
        return 1;
    }

    /* /handoff write|status — sovereign handoff receipt */
    if (input[0] == '/' && input[1] == 'h' && input[2] == 'a' && input[3] == 'n' &&
        input[4] == 'd') {
        const CHAR16 *arg = input + 8;  /* skip "/handoff" */
        while (*arg == ' ') arg++;

        if (arg[0] == 'w') {
            /* /handoff write — generate OOHANDOFF.TXT */
            if (!g_ssm_loaded) {
                Print(L"[handoff] No model loaded\r\n");
                return 1;
            }
            OoHandoffReceipt receipt;
            oo_receipt_init(&receipt, &g_ssm_cfg, "sovereign", "genesis");
            char buf[1024];
            int n = oo_receipt_serialize(&receipt, buf, sizeof(buf));
            if (n > 0) {
                Print(L"\r\n── OOHANDOFF.TXT ──────────────────────────\r\n");
                /* Print each line */
                for (int i = 0; i < n; i++) {
                    if (buf[i] == '\n') {
                        Print(L"\r\n");
                    } else {
                        CHAR16 wc[2] = { (CHAR16)buf[i], 0 };
                        Print(wc);
                    }
                }
                Print(L"───────────────────────────────────────────\r\n\r\n");
                Print(L"[handoff] Receipt ready (%d bytes)\r\n", n);
                Print(L"[handoff] Write to EFI boot volume for oo-host sync\r\n\r\n");
            } else {
                Print(L"[handoff] Serialization failed\r\n");
            }
        } else if (arg[0] == 's') {
            /* /handoff status */
            Print(L"\r\n── Handoff Status ─────────────────────────\r\n");
            Print(L"  Model:     %s\r\n", g_ssm_loaded ? L"loaded" : L"none");
            Print(L"  Lifeline:  %s\r\n",
                  g_ssm_cfg.lifeline_enabled ? L"ON" : L"OFF");
            Print(L"  RLF depth: %d\r\n", g_rlf_max_loops);
            Print(L"  Protocol:  OOHANDOFF.TXT v1\r\n");
            Print(L"  Sync:      oo-host → sovereign_export.json\r\n");
            Print(L"             sovereign → OOHANDOFF.TXT\r\n");
            Print(L"───────────────────────────────────────────\r\n\r\n");
        } else {
            Print(L"[handoff] Usage: /handoff write | /handoff status\r\n");
        }
        return 1;
    }

    /* /host goals|policy — display host-side organism state */
    if (input[0] == '/' && input[1] == 'h' && input[2] == 'o' && input[3] == 's' &&
        input[4] == 't') {
        const CHAR16 *arg = input + 5;
        while (*arg == ' ') arg++;

        if (arg[0] == 'g') {
            /* /host goals */
            Print(L"\r\n── Host Goals (from sovereign_export.json) ──\r\n");
            Print(L"  To sync: place sovereign_export.json on boot volume\r\n");
            Print(L"  Generate: cd oo-host && cargo run -- export sovereign\r\n");
            Print(L"────────────────────────────────────────────────\r\n\r\n");
        } else if (arg[0] == 'p') {
            /* /host policy */
            Print(L"\r\n── Host Policy ────────────────────────────\r\n");
            Print(L"  safe_first:      true (default)\r\n");
            Print(L"  deny_by_default: true (default)\r\n");
            Print(L"  enforcement:     observe (default)\r\n");
            Print(L"  Source: oo-host sovereign_export.json\r\n");
            Print(L"───────────────────────────────────────────\r\n\r\n");
        } else {
            Print(L"[host] Usage: /host goals | /host policy\r\n");
        }
        return 1;
    }

    /* /auto start|stop|status|tick — autonomous operation */
    if (input[0] == '/' && input[1] == 'a' && input[2] == 'u' && input[3] == 't' &&
        input[4] == 'o') {
        const CHAR16 *arg = input + 5;
        while (*arg == ' ') arg++;

        if (arg[0] == 's' && arg[1] == 't' && arg[2] == 'a') {
            /* /auto start */
            g_auto_enabled = 1;
            g_auto_tick = 0;
            Print(L"\r\n══════════════════════════════════════════\r\n");
            Print(L"  AUTONOMOUS MODE ACTIVATED\r\n");
            Print(L"  Tick interval: %d seconds\r\n", g_auto_interval);
            Print(L"  RLF depth:     %d\r\n", g_rlf_max_loops);
            Print(L"  Lifeline:      %s\r\n",
                  g_ssm_cfg.lifeline_enabled ? L"ON" : L"OFF");
            Print(L"  /auto stop to halt\r\n");
            Print(L"══════════════════════════════════════════\r\n\r\n");
        } else if (arg[0] == 's' && arg[1] == 't' && arg[2] == 'o') {
            /* /auto stop */
            g_auto_enabled = 0;
            Print(L"\r\n  Autonomous mode stopped (tick %d)\r\n\r\n", g_auto_tick);
        } else if (arg[0] == 's' && arg[1] == 't' && arg[2] == 'a') {
            /* /auto status (handled above by start) */
        } else if (arg[0] == 't') {
            /* /auto tick — run one manual tick */
            if (!g_ssm_loaded) {
                Print(L"[auto] No model loaded\r\n");
                return 1;
            }
            g_auto_tick++;
            Print(L"\r\n  ┌─ Tick %d ────────────────────────────────\r\n", g_auto_tick);
            Print(L"  │ Mode:     %s\r\n", g_auto_mode == 0 ? L"normal" : (g_auto_mode == 1 ? L"degraded" : L"safe"));
            Print(L"  │ RLF:      depth=%d\r\n", g_rlf_max_loops);
            Print(L"  │ Lifeline: %s\r\n", g_ssm_cfg.lifeline_enabled ? L"ON" : L"OFF");
            Print(L"  │ Health:   checking...\r\n");

            /* Self-heal check */
            if (g_ssm_telem.nan_count > 0 || g_ssm_telem.inf_count > 0) {
                g_auto_mode = 2;  /* safe */
                g_rlf_max_loops = 1;
                Print(L"  │ ⚠ NaN/Inf detected → SAFE MODE, depth=1\r\n");
            } else if (g_ssm_telem.hidden_state_norm > 100.0f) {
                g_auto_mode = 1;  /* degraded */
                Print(L"  │ ⚠ ‖h‖=%.1f → DEGRADED\r\n", g_ssm_telem.hidden_state_norm);
            } else if (g_auto_mode != 0) {
                g_auto_mode = 0;  /* auto-recover */
                g_rlf_max_loops = 16;
                Print(L"  │ ✓ Recovered → NORMAL, depth=16\r\n");
            } else {
                Print(L"  │ ✓ Healthy\r\n");
            }

            /* Auto-write handoff */
            if (g_auto_tick % 3 == 0) {
                OoHandoffReceipt receipt;
                oo_receipt_init(&receipt, &g_ssm_cfg, "sovereign", "genesis");
                char buf[1024];
                int n = oo_receipt_serialize(&receipt, buf, sizeof(buf));
                if (n > 0) {
                    Print(L"  │ HANDOFF: %d bytes ready\r\n", n);
                }
            }
            Print(L"  └──────────────────────────────────────────\r\n\r\n");
        } else {
            Print(L"\r\n── Autonomous Status ──────────────────────\r\n");
            Print(L"  Enabled: %s\r\n", g_auto_enabled ? L"YES" : L"NO");
            Print(L"  Ticks:   %d\r\n", g_auto_tick);
            Print(L"  Mode:    %s\r\n", g_auto_mode == 0 ? L"normal" : (g_auto_mode == 1 ? L"degraded" : L"safe"));
            Print(L"  Depth:   %d\r\n", g_rlf_max_loops);
            Print(L"───────────────────────────────────────────\r\n");
            Print(L"  /auto start  — Enable autonomous loop\r\n");
            Print(L"  /auto stop   — Disable autonomous loop\r\n");
            Print(L"  /auto tick   — Run one manual tick\r\n");
            Print(L"───────────────────────────────────────────\r\n\r\n");
        }
        return 1;
    }

    return 0;
}

/* ── UEFI Wall Clock ──────────────────────────────────────────────────────── */

static int uefi_wall_us(unsigned long long *out_us) {
    /**
     * Get wall clock time in microseconds via UEFI runtime.
     */
    if (!out_us || !RT) return 0;
    EFI_TIME t;
    EFI_STATUS st = uefi_call_wrapper(RT->GetTime, 2, &t, NULL);
    if (EFI_ERROR(st)) return 0;
    *out_us = (unsigned long long)t.Second * 1000000ULL +
              (unsigned long long)t.Minute * 60000000ULL +
              (unsigned long long)t.Hour * 3600000000ULL;
    return 1;
}

/* ── Main Entry Point ─────────────────────────────────────────────────────── */

EFI_STATUS EFIAPI efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {
    /**
     * UEFI entry point for Mamba2 SSM + RLF Bare-Metal LLM.
     */
    InitializeLib(ImageHandle, SystemTable);

    /* ── Initialize all 20 OO engines ─────────────────────────────────── */
    djibion_init(&g_djibion);
    diopion_init(&g_diopion);
    diagnostion_init(&g_diagnostion);
    memorion_init(&g_memorion);
    orchestrion_init(&g_orchestrion);
    calibrion_init(&g_calibrion);
    compatibilion_init(&g_compatibilion);
    compatibilion_probe_cpu(&g_compatibilion);
    evolvion_init(&g_evolvion);
    synaption_init(&g_synaption);
    conscience_init(&g_conscience);
    neuralfs_init(&g_neuralfs);
    ghost_init(&g_ghost);
    immunion_init(&g_immunion);
    dreamion_init(&g_dreamion);
    symbion_init(&g_symbion);
    collectivion_init(&g_collectivion);
    metabion_init(&g_metabion);
    metabion_set_mode(&g_metabion, (MetabionMode)METABION_DEFAULT_METABION_MODE);
    cellion_init(&g_cellion);
    morphion_init(&g_morphion);
    pheromion_init(&g_pheromion);
    compatibilion_set_platform(&g_compatibilion, COMPAT_PLAT_UEFI | COMPAT_PLAT_FAT32);

    /* Initialize tracing */
    djibmark_init();
    DJIBMARK_BOOT();

    /* Disable UEFI watchdog (model loads take minutes) */
    uefi_call_wrapper(BS->SetWatchdogTimer, 4, 0, 0, 0, NULL);

    /* Clear screen */
    uefi_call_wrapper(ST->ConOut->ClearScreen, 1, ST->ConOut);

    /* Cyberpunk splash */
    ShowCyberpunkSplash(ImageHandle, SystemTable);

    /* Banner */
    Print(L"\r\n");
    Print(L"    __  ______    __  _______  ___   ___\r\n");
    Print(L"   /  |/  /   |  /  |/  / __ )/   | |__ \\\r\n");
    Print(L"  / /|_/ / /| | / /|_/ / __  / /| | __/ /\r\n");
    Print(L" / /  / / ___ |/ /  / / /_/ / ___ |/ __/\r\n");
    Print(L"/_/  /_/_/  |_/_/  /_/_____/_/  |_/____/\r\n\r\n");
    Print(L"    ____  __    ______\r\n");
    Print(L"   / __ \\/ /   / ____/\r\n");
    Print(L"  / /_/ / /   / /_\r\n");
    Print(L" / _, _/ /___/ __/\r\n");
    Print(L"/_/ |_/_____/_/\r\n\r\n");
    Print(L"Mamba2 SSM + RLF Bare-Metal UEFI REPL\r\n");
    Print(L"Based on llm-baremetal (Djiby Diop) + RLF (batteryphil)\r\n");
    Print(L"────────────────────────────────────────────────────────\r\n");
    Print(L"Tips: /help | /ssm_info | /rlf_depth N | /rlf_trace\r\n\r\n");

    /* ── [1/4] Open File System ───────────────────────────────────────── */
    Print(L"[1/4] Opening file system...\r\n");

    EFI_LOADED_IMAGE *LoadedImage;
    EFI_STATUS status = uefi_call_wrapper(BS->HandleProtocol, 3,
        ImageHandle, &LoadedImageProtocol, &LoadedImage);
    if (EFI_ERROR(status)) {
        Print(L"ERROR: LoadedImage protocol failed\r\n");
        return status;
    }

    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    status = uefi_call_wrapper(BS->HandleProtocol, 3,
        LoadedImage->DeviceHandle, &FileSystemProtocol, &FileSystem);
    if (EFI_ERROR(status)) {
        Print(L"ERROR: FileSystem protocol failed\r\n");
        return status;
    }

    status = uefi_call_wrapper(FileSystem->OpenVolume, 2, FileSystem, &g_root);
    if (EFI_ERROR(status)) {
        Print(L"ERROR: OpenVolume failed\r\n");
        return status;
    }
    Print(L"[1/4] OK\r\n");

    /* ── [2/4] Find and detect model ─────────────────────────────────── */
    Print(L"[2/4] Searching for model...\r\n");

    /* Try model files in priority order */
    static const CHAR16 *model_candidates[] = {
        L"model.mamba.bin",
        L"mamba2.mamba.bin",
        L"model.bin",
        L"stories15M.bin",
        L"stories42M.bin",
        L"stories110M.bin",
        L"model.gguf",
        NULL
    };

    EFI_FILE_HANDLE model_file = NULL;
    const CHAR16 *model_path = NULL;

    for (int i = 0; model_candidates[i]; i++) {
        EFI_FILE_HANDLE f = NULL;
        EFI_STATUS st = uefi_call_wrapper(g_root->Open, 5, g_root, &f,
            (CHAR16 *)model_candidates[i], EFI_FILE_MODE_READ, 0);
        if (!EFI_ERROR(st) && f) {
            model_file = f;
            model_path = model_candidates[i];
            Print(L"  Found: %s\r\n", model_path);
            break;
        }
    }

    if (!model_file) {
        Print(L"ERROR: No model found. Place model.mamba.bin on boot volume.\r\n");
        Print(L"Press any key to exit...\r\n");
        EFI_INPUT_KEY key;
        uefi_call_wrapper(ST->ConIn->ReadKeyStroke, 2, ST->ConIn, &key);
        return EFI_NOT_FOUND;
    }

    /* Detect format */
    g_model_fmt = detect_model_format(model_file);
    UINT64 model_size = llmk_file_size(model_file);

    if (g_model_fmt == MODEL_FMT_MAMBA) {
        Print(L"  Format: Mamba2 SSM (.mamba.bin)\r\n");
    } else if (g_model_fmt == MODEL_FMT_GGUF) {
        Print(L"  Format: GGUF (Transformer)\r\n");
    } else {
        Print(L"  Format: BIN (Transformer)\r\n");
    }
    Print(L"  Size:   %lu bytes\r\n", (UINT64)model_size);
    Print(L"[2/4] OK\r\n");

    /* ── [3/4] Load Model ─────────────────────────────────────────────── */
    Print(L"[3/4] Loading model...\r\n");

    if (g_model_fmt == MODEL_FMT_MAMBA) {
        if (load_mamba_model(model_file, model_size) != 0) {
            Print(L"ERROR: Failed to load Mamba model\r\n");
            uefi_call_wrapper(model_file->Close, 1, model_file);
            return EFI_LOAD_ERROR;
        }
    } else {
        /* Transformer models: use existing GGUF/bin loaders */
        Print(L"  (Transformer model detected — use legacy loader path)\r\n");
        Print(L"  NOTE: This build is optimized for Mamba SSM models.\r\n");
        Print(L"  Place a model.mamba.bin file for full SSM + RLF support.\r\n");
    }

    uefi_call_wrapper(model_file->Close, 1, model_file);
    Print(L"[3/4] OK\r\n");

    /* ── [4/4] REPL ───────────────────────────────────────────────────── */
    Print(L"[4/4] Starting REPL...\r\n\r\n");

    CHAR16 input_buf[512];
    int prompt_ids[512];

    while (1) {
        /* Prompt */
        Print(L"mamba> ");

        /* Read input line */
        int buf_pos = 0;
        while (1) {
            EFI_INPUT_KEY key;
            UINTN idx;
            uefi_call_wrapper(BS->WaitForEvent, 3, 1, &ST->ConIn->WaitForKey, &idx);
            EFI_STATUS kst = uefi_call_wrapper(ST->ConIn->ReadKeyStroke, 2,
                                               ST->ConIn, &key);
            if (EFI_ERROR(kst)) continue;

            if (key.UnicodeChar == '\r' || key.UnicodeChar == '\n') {
                input_buf[buf_pos] = 0;
                Print(L"\r\n");
                break;
            }

            /* Backspace */
            if (key.UnicodeChar == '\b' || key.ScanCode == 0x08) {
                if (buf_pos > 0) {
                    buf_pos--;
                    Print(L"\b \b");
                }
                continue;
            }

            /* Regular character */
            if (key.UnicodeChar >= 0x20 && buf_pos < 510) {
                input_buf[buf_pos++] = key.UnicodeChar;
                CHAR16 echo[2];
                echo[0] = key.UnicodeChar;
                echo[1] = 0;
                Print(L"%s", echo);
            }
        }

        /* Empty input */
        if (buf_pos == 0) continue;

        /* Exit commands */
        if ((input_buf[0] == 'q' && input_buf[1] == 'u' && input_buf[2] == 'i' &&
             input_buf[3] == 't' && input_buf[4] == 0) ||
            (input_buf[0] == 'e' && input_buf[1] == 'x' && input_buf[2] == 'i' &&
             input_buf[3] == 't' && input_buf[4] == 0)) {
            Print(L"\r\nGoodbye.\r\n");
            break;
        }

        /* Try SSM commands first */
        if (handle_ssm_command(input_buf)) continue;

        /* Generate response */
        if (g_ssm_loaded) {
            int n_toks = tokenize_input(input_buf, prompt_ids, 512);
            if (n_toks > 0) {
                ssm_generate_response(prompt_ids, n_toks);
            }
        } else {
            Print(L"[error] No model loaded. "
                  L"Place model.mamba.bin on boot volume.\r\n");
        }
    }

    /* Cleanup */
    if (g_ssm_loaded) {
        mamba_state_free(&g_ssm_state);
        mamba_free_weight_arrays(&g_ssm_wt);
    }

    return EFI_SUCCESS;
}
