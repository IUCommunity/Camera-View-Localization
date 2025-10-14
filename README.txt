# Fast mode for quick results
python localization.py --camera view1.png --map map1.png --mode fast

# Balanced mode (default) - best overall performance
python localization.py --camera view1.png --map map1.png --mode balanced

# High accuracy mode for critical applications
python localization.py --camera view1.png --map map1.png --mode high_accuracy

# JSON output only for integration
python localization.py --camera view1.png --map map1.png --mode fast --json

# Custom parameters
python localization.py --camera view1.png --map map1.png --mode balanced --samples 2 --max-size 512