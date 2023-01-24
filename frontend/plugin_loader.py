import os

from backend.singleton import singleton
import importlib
gs = singleton


class PluginLoader():
    def __init__(self, parent):
        self.parent = parent
        gs.plugins = {}


    def list_plugins(self):
        #path = gs.system.plugins
        path = 'plugins'
        # Full paths would load with f.path
        plugins = [f.name for f in os.scandir(path) if f.is_dir()]
        if '__pycache__' in plugins:
            plugins.remove('__pycache__')
        return plugins

    def load_plugin(self, plugin_name):
        module = importlib.import_module(plugin_name)
        gs.plugins[plugin_name] = module.aiNodesPlugin(self.parent)

        globals().update(
            {n: getattr(gs.plugins[plugin_name], n) for n in gs.plugins[plugin_name].__all__} if hasattr(gs.plugins[plugin_name], '__all__')
            else
            {k: v for (k, v) in gs.plugins[plugin_name].__dict__.items() if not k.startswith('_')
        })

        gs.plugins[plugin_name].initme()
    def unload_plugin(self, plugin_name):
        gs.plugins[plugin_name] = None
        del gs.plugins[plugin_name]
        try:
            gs.plugins[plugin_name].initme()
        except Exception as e:
            print(f"{plugin_name} plugin unloaded")
