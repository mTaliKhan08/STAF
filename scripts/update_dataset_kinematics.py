path = r"E:\MS-DS\Thesis-1\Multi-Modal Fusion Framework for Autonomous Robot\STAF\staf\data\dataset.py"

with open(path, "r", encoding="utf-8") as f:
    c = f.read()

old_load_wp_end = '''        # Transform high-level route navigation goal into ego body frame
        target_point = transform_world_point_to_ego(cur_M, (x_cmd, y_cmd))
        spd = float(cur_data.get("speed", 0.0))
        sample_info["steer"] = float(cur_data.get("steer", 0.0))

        return np.array(waypoints, dtype=np.float32), np.array(target_point, dtype=np.float32), spd'''

new_load_wp_end = '''        # Transform high-level route navigation goal into ego body frame
        target_point = transform_world_point_to_ego(cur_M, (x_cmd, y_cmd))
        spd = float(cur_data.get("speed", 0.0))
        sample_info["steer"] = float(cur_data.get("steer", 0.0))

        # Causal kinematics from frame t and t-1
        ax = 0.0
        body_w = 0.0
        if idx > 0 and len(frames) > idx:
            prev_meas_path = os.path.join(meas_dir, f"{frames[idx - 1]}.json")
            if os.path.exists(prev_meas_path):
                try:
                    with open(prev_meas_path, "r") as pf:
                        d_prev = json.load(pf)
                    spd_prev = float(d_prev.get("speed", spd))
                    ax = (spd - spd_prev) / 0.5
                    th_cur = float(cur_data.get("theta", 0.0))
                    th_prev = float(d_prev.get("theta", th_cur))
                    d_th = wrap_angle(th_cur - th_prev)
                    body_w = -d_th / 0.5
                except Exception:
                    pass

        sample_info["accel"] = float(ax)
        sample_info["body_yaw_rate"] = float(body_w)
        sample_info["junction"] = float(bool(cur_data.get("junction", False)))
        sample_info["vehicle_hazard"] = float(bool(cur_data.get("vehicle_hazard", False)))
        sample_info["light_hazard"] = float(bool(cur_data.get("light_hazard", False)))
        sample_info["stop_sign_hazard"] = float(bool(cur_data.get("stop_sign_hazard", False)))

        return np.array(waypoints, dtype=np.float32), np.array(target_point, dtype=np.float32), spd'''

assert old_load_wp_end in c, "old_load_wp_end not found"
c = c.replace(old_load_wp_end, new_load_wp_end)

old_getitem_end = '''        sample_dict["waypoints"] = torch.from_numpy(wps).float()
        sample_dict["target_point"] = torch.from_numpy(target).float()
        sample_dict["speed"] = torch.tensor([spd], dtype=torch.float32)
        sample_dict["steer"] = torch.tensor([sample_info.get("steer", 0.0)], dtype=torch.float32)
        return sample_dict'''

new_getitem_end = '''        sample_dict["waypoints"] = torch.from_numpy(wps).float()
        sample_dict["target_point"] = torch.from_numpy(target).float()
        sample_dict["target_point_norm"] = torch.clamp(sample_dict["target_point"] / 20.0, -3.0, 3.0)
        sample_dict["speed"] = torch.tensor([spd], dtype=torch.float32)
        sample_dict["steer"] = torch.tensor([sample_info.get("steer", 0.0)], dtype=torch.float32)
        sample_dict["accel"] = torch.tensor([sample_info.get("accel", 0.0)], dtype=torch.float32)
        sample_dict["body_yaw_rate"] = torch.tensor([sample_info.get("body_yaw_rate", 0.0)], dtype=torch.float32)
        sample_dict["x_kinematic_3d"] = torch.tensor([spd / 10.0, sample_info.get("accel", 0.0), sample_info.get("body_yaw_rate", 0.0)], dtype=torch.float32)
        sample_dict["junction"] = torch.tensor([sample_info.get("junction", 0.0)], dtype=torch.float32)
        sample_dict["vehicle_hazard"] = torch.tensor([sample_info.get("vehicle_hazard", 0.0)], dtype=torch.float32)
        sample_dict["light_hazard"] = torch.tensor([sample_info.get("light_hazard", 0.0)], dtype=torch.float32)
        sample_dict["stop_sign_hazard"] = torch.tensor([sample_info.get("stop_sign_hazard", 0.0)], dtype=torch.float32)
        sample_dict["route_dir"] = sample_info.get("route_dir", "")
        return sample_dict'''

assert old_getitem_end in c, "old_getitem_end not found"
c = c.replace(old_getitem_end, new_getitem_end)

with open(path, "w", encoding="utf-8") as f:
    f.write(c)

print("dataset.py updated with causal kinematics and hazard metadata successfully!")

