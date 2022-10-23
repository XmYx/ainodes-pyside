import random, os, json



# nsp_parse( prompt )
# Input: dict, list, str
# Parse strings for terminology keys and replace them with random terms

class parser:

	def __init__(self):
		self.nspterminology = None
		self.get_data()


	def wget(self,url, output):
		import subprocess
		res = subprocess.run(['wget', '-q', url, '-O', output], stdout=subprocess.PIPE).stdout.decode('utf-8')
		print(res)


	def get_data(self):

		if not os.path.exists('./sd/nsp/nsp_pantry.json'):
			self.wget('https://raw.githubusercontent.com/WASasquatch/noodle-soup-prompts/main/nsp_pantry.json',
				 './scripts/ui/nsp/nsp_pantry.json')

		if self.nspterminology is None:
			with open('./sd/nsp/nsp_pantry.json', 'r', encoding="utf-8") as f:
				self.nspterminology = json.loads(f.read())


	def parse(self, prompt):

		new_prompt = ''
		new_prompts = []
		new_dict = {}
		ptype = type(prompt)

		self.get_data()

		if ptype == dict:
			for pstep, pvalue in prompt.items():
				if type(pvalue) == list:
					for prompt in pvalue:
						new_prompt = f'_{prompt}_'
						for term in self.nspterminology:
							tkey = f'_{term}_'
							tcount = prompt.count(tkey)
							for i in range(tcount):
								new_prompt = new_prompt.replace(tkey, random.choice(self.nspterminology[term]), 1)
						new_prompts.append(new_prompt)
					new_dict[pstep] = new_prompts
					new_prompts = []
			return new_dict
		elif ptype == list:
			for pstr in prompt:
				new_prompt = f'_{pstr}_'
				for term in self.nspterminology:
					tkey = f'_{term}_'
					tcount = new_prompt.count(tkey)
					for i in range(tcount):
						new_prompt = new_prompt.replace(tkey, random.choice(self.nspterminology[term]), 1)
				new_prompts.append(new_prompt)
				new_prompt = None
			return ', '.join(new_prompts)
		elif ptype == str:
			new_prompt = f'_{prompt}_'
			for term in self.nspterminology:
				tkey = f'_{term}_'
				tcount = new_prompt.count(tkey)
				for i in range(tcount):
					new_prompt = new_prompt.replace(tkey, random.choice(self.nspterminology[term]), 1)
			return new_prompt
		else:
			return


	def get_nsp_keys(self):
		return list(self.nspterminology.keys())
