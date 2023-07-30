import inspect
import pytorchvideo.losses

for name, obj in inspect.getmembers(pytorchvideo):
        print(obj)