try:
    import gc
    import torch


    def torch_gc():
        from backend.hardware import empty_cache
        empty_cache()
except:
    def torch_gc():
        pass
