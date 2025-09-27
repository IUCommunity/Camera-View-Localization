For Maximum Speed (Testing)
python main.py --camera view1.png --map map1.png --fast --max-tiles 5 --max-image-size 1024

For Balanced Speed/Quality (Production)
python main.py --camera view1.png --map map1.png --early-termination 0.8 --samples 2

For Large Maps (Quick Results)
python main.py --camera view1.png --map map1.png --max-tiles 10 --stride 1024

For High-Resolution Images
python main.py --camera view1.png --map map1.png --max-image-size 2048 --fast