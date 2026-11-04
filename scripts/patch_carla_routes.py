with open("../../STAF/staf/data/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

old_block = """        # Find all routes (either in subfolders like dirt_dataset_23_11 or directly)
        routes = []
        entries = sorted(os.listdir(self.data_root))
        for entry in entries:
            p = os.path.join(self.data_root, entry)
            if not os.path.isdir(p):
                continue
            # Check if this is a scenario directory containing routes
            sub_entries = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if any("route" in d.lower() or "routes" in d.lower() for d in sub_entries):
                for sub in sub_entries:
                    route_path = os.path.join(p, sub)
                    if os.path.isdir(os.path.join(route_path, "rgb")):
                        routes.append(route_path)
            elif os.path.isdir(os.path.join(p, "rgb")):
                routes.append(p)"""

new_block = """        # Recursively find all route directories containing sensor recordings
        routes = []
        for root, dirs, files in os.walk(self.data_root):
            if "rgb" in dirs:
                routes.append(root)
        routes = sorted(list(set(routes)))"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open("../../STAF/staf/data/dataset.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("Successfully patched CARLASTAFDataset route discovery with recursive os.walk!")
else:
    print("ERROR: old_block not found in dataset.py")

