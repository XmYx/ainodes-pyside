import threading

defaults = {}
txt2img = {}
vid2vid = {}
txt2vid = {}
img2img = {}
models = {}
current_images = []
current_video_frames = []
current_video = ''
rendering = False
render_mode = ''
device = ''
modelFS = ''
modelCS = ''

class Singleton:
	_instance = None
	_lock = threading.Lock()

	def __new__(cls, *args, **kwargs):
		if not cls._instance:
			with cls._lock:
				# another thread could have created the instance
				# before we acquired the lock. So check that the
				# instance is still nonexistent.
				if not cls._instance:
					cls._instance = super(Singleton, cls).__new__(cls)
		return cls._instance
