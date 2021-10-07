
import sys
import copy

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

args = sys.argv[1:]
assert len(args) == 2
my_computer = int(args[0])
num_computers = int(args[1])

print(f"MY_COMPUTER: {my_computer},  NUM_COMPUTERS: {num_computers}")

script = "rlpyt/ul/experiments/rl_with_ul/scripts/maze/train/maze_dqn_with_ul_serial.py"

affinity_code = quick_affinity_code(contexts_per_gpu=1, use_gpu=True)
runs_per_setting = 2
experiment_title = "maze_dqn_with_ul_schedule_1"

variant_levels_1 = list()
# variant_levels_2 = list()
# variant_levels_3 = list()

#stop_conv_grads = [False, True]
stop_conv_grads = [False]
values = list(zip(stop_conv_grads))
dir_names = ["{}stpcnvgrd".format(*v) for v in values]
keys = [("model", "stop_conv_grad")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

min_steps_rl = [5e3]
min_steps_ul = [1e3]
ul_lr_schedule = ["cosine"]
values = list(zip(min_steps_rl, min_steps_ul, ul_lr_schedule))
dir_names = ["{}rlminstepsul{}_{}anneal".format(*v) for v in values]
keys = [("algo", "min_steps_rl"), ("algo", "min_steps_ul"), ("algo", "ul_lr_schedule")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

ul_update_schedules = [
    "constant_1",
#    "linear_2",
#    "quadratic_3",
#    "mod_2",
] 
values = list(zip(ul_update_schedules))
dir_names = ["{}".format(*v) for v in values]
keys = [("algo", "ul_update_schedule")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

# games = ["pong", "qbert", "seaquest", "space_invaders",
#     "alien", "breakout", "frostbite", "gravitar"]
# games = ["breakout", "gravitar", "qbert", "space_invaders"]
# values = list(zip(games))
# dir_names = games
# keys = [("env", "game")]
# variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))
# variant_levels_3.append(VariantLevel(keys, values, dir_names))

##################################################
# RL CONFIG (mostly)


variants_1, log_dirs_1 = make_variants(*variant_levels_1)
print(variants_1, log_dirs_1)
# variants_2, log_dirs_2 = make_variants(*variant_levels_2)

variants = variants_1  # + variants_2
log_dirs = log_dirs_1  # + log_dirs_2

num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]

default_config_key = "scaled_ddqn_ul"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)

