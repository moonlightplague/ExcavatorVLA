# ExcavatorVLA

## ZSP Isaac Sim Entry

Use [main_zsp.py](main_zsp.py) to launch the promoted excavator/sand runtime. [main.py](main.py) is a thin compatibility wrapper that runs [main_zsp.py](main_zsp.py). The launcher can load either the flat ZSP USD bundle in [assets/zsp](assets/zsp) or the original textured scene in [assets/usd/excavator_scene.usd](assets/usd/excavator_scene.usd).

[main_zsp.py](main_zsp.py) resolves the repo root, adds [scripts](scripts) to the Python import path, opens the selected model scene, and then runs `excavator_app.bootstrap.run_excavator_with_sand()`.

Choose the model source in [excavator_config.json](excavator_config.json):

```json
{
	"model_source": "zsp",
	"project_root": "/isaac-sim/ExcavatorVLA"
}
```

Use `"model_source": "zsp"` to load [assets/zsp/URDF_real3.usd](assets/zsp/URDF_real3.usd). For this mode, [main_zsp.py](main_zsp.py) also normalizes old `configuration/*.usd` sublayer references to the flat [assets/zsp](assets/zsp) layout.

Use `"model_source": "original"` to load the original textured scene [assets/usd/excavator_scene.usd](assets/usd/excavator_scene.usd), which is the scene used by [run_excavator_standalone.py](run_excavator_standalone.py).

Run this one-liner in Isaac Sim Script Editor:

```python
import runpy

runpy.run_path("/isaac-sim/ExcavatorVLA/main_zsp.py", run_name="__main__")
```

You can also run `main.py` from the repository root; it forwards to `main_zsp.py`.

Environment variables still work as temporary overrides: set `EXCAVATOR_PROJECT_ROOT` or `EXCAVATOR_USE_ZSP_MODELS` before running [main_zsp.py](main_zsp.py) if you do not want to edit the config file.

The expected ZSP USD files are:

- [assets/zsp/URDF_real3.usd](assets/zsp/URDF_real3.usd)
- [assets/zsp/URDF_real3_base.usd](assets/zsp/URDF_real3_base.usd)
- [assets/zsp/URDF_real3_physics.usd](assets/zsp/URDF_real3_physics.usd)
- [assets/zsp/URDF_real3_robot.usd](assets/zsp/URDF_real3_robot.usd)
- [assets/zsp/URDF_real3_sensor.usd](assets/zsp/URDF_real3_sensor.usd)
