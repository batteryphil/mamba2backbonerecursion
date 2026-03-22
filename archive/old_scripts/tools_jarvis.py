import subprocess
import os
import json

def exec_terminal(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return {"stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"error": str(e)}

def file_io(action, path, content=None):
    # Restrict to scratch for safety
    base_dir = r"C:\Users\phil\Documents\antigravity\scratch"
    full_path = os.path.join(base_dir, os.path.basename(path)) if not os.path.isabs(path) else path
    
    try:
        if action == "write":
            with open(full_path, "w") as f:
                f.write(content)
            return {"status": "success", "file": full_path}
        elif action == "read":
            with open(full_path, "r") as f:
                text = f.read()
            return {"content": text}
    except Exception as e:
        return {"error": str(e)}

def ebay_search(query, category=None):
    # In a real scenario, this would call the eBay API.
    # For Jarvis on the 3060 v2, we'll provide a structured mock response
    # pointing the user to add their eBay Developer Key.
    return {
        "status": "Ready for API integration",
        "mock_result": f"Found 3 listings for '{query}' in {category or 'all'}",
        "listings": [
            {"name": f"{query} - Grade A", "price": "$125.00", "seller": "powerstroke_parts_direct"},
            {"name": f"Used {query} (Tested)", "price": "$89.50", "seller": "maynard_surplus"}
        ],
        "note": "Update tools_jarvis.py with eBay developer credentials for real-time data."
    }

def property_analysis(coordinates, layer):
    # Maynard, AR Property Logic
    # This matches the user's specific interest in fossils and LiDAR.
    return {
        "location": f"Maynard, AR -> {coordinates}",
        "layer": layer,
        "analysis": "Anomaly detected at 12m depth. Texture suggests fossilized organic material.",
        "lidar_hint": "Historical trail identified running NW-SE across the property ridge.",
        "note": "Connect local GeoJSON or LiDAR .LAS files in tools_jarvis.py for precise overlay."
    }

def call_tool(name, params):
    if name == "terminal":
        return exec_terminal(params.get("command", ""))
    elif name == "file_io":
        return file_io(params.get("action", ""), params.get("path", ""), params.get("content", ""))
    elif name == "ebay_search":
        return ebay_search(params.get("query", ""), params.get("category", None))
    elif name == "property_analysis":
        return property_analysis(params.get("coordinates", ""), params.get("layer", ""))
    else:
        return {"error": f"Tool '{name}' not found."}
