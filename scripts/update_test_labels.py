test_path = r"E:\MS-DS\Thesis-1\Multi-Modal Fusion Framework for Autonomous Robot\STAF\tests\test_labels.py"
with open(test_path, "r", encoding="utf-8") as f:
    c = f.read()

# Fix the call in __main__
c = c.replace("test_kinematics_causal(routes)", "test_kinematics_causal()")

with open(test_path, "w", encoding="utf-8") as f:
    f.write(c)

print("Fixed call to test_kinematics_causal")
