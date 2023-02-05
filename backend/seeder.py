import os
import random
import secrets

# here we try to make random as random is possible. The pseudo-random number generator is
# kind of weak as it might start repeating itself during one session.
# to prevent this we seed it anytime we request a strong cryptographic seed

def get_strong_seed(max):
    rnd = secrets.randbelow(max)
    random.seed(secrets.randbelow(rnd))
    os.environ["PL_GLOBAL_SEED"] = str(rnd)
    return rnd

def get_strong_seed_range(min, max):
    rnd = min + secrets.randbelow(max - min + 1)
    random.seed(secrets.randbelow(rnd))
    os.environ["PL_GLOBAL_SEED"] = str(rnd)
    return rnd