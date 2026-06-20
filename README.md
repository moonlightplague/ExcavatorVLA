# ExcavatorVLA

## ZSP Isaac Sim Entry

Use [main_zsp.py](main_zsp.py) to launch the promoted excavator/sand runtime against the flat USD bundle in [assets/zsp](assets/zsp).

[main_zsp.py](main_zsp.py) resolves the repo root, adds [scripts](scripts) to the Python import path, normalizes old `configuration/*.usd` sublayer references to the flat [assets/zsp](assets/zsp) layout, opens [assets/zsp/URDF_real3.usd](assets/zsp/URDF_real3.usd), and then runs `excavator_app.bootstrap.run_excavator_with_sand()`.

The expected ZSP USD files are:

- [assets/zsp/URDF_real3.usd](assets/zsp/URDF_real3.usd)
- [assets/zsp/URDF_real3_base.usd](assets/zsp/URDF_real3_base.usd)
- [assets/zsp/URDF_real3_physics.usd](assets/zsp/URDF_real3_physics.usd)
- [assets/zsp/URDF_real3_robot.usd](assets/zsp/URDF_real3_robot.usd)
- [assets/zsp/URDF_real3_sensor.usd](assets/zsp/URDF_real3_sensor.usd)