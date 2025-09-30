     for flag, value in {**common, **extra}.items():
         if value is None:
             continue
        cmd.append(f"{flag}={value}")
     if extra_args:
         for item in extra_args:
             if "=" in item:
                 flag, value = item.split("=", 1)
                cmd.append(f"{flag}={value}")
             else:
                 cmd.append(item)
     return cmd
     try:
         import matplotlib.pyplot as plt
     except ImportError as exc:  # pragma: no cover - plotting dependency optional
        print("matplotlib not installed; skipping plot generation.")
        return
 
     if not runs:
         print("No runs to plot.")
