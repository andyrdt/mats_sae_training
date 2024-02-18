# %%
from neel.imports import *
from neel_plotly import *

# %%
import wandb

entity = "andyrdt"
project = "othello_gpt_sae"

for artifact in [
    # SAEs trained over all seq positions:
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512:v0",
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024:v0",
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_2048:v3",
    # "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_4096:v4",
    # SAEs trained excluding first seq position:
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512:v2",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024:v2",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_2048:v5",
    "sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_4096:v6",
]:
    artifact_path = f"{entity}/{project}/{artifact}"
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact.download()

# %%
from sae_training.utils import LMSparseAutoencoderSessionloader

# path ="./artifacts/sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512:v2/final_sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_512.pt"
path = "./artifacts/sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024:v2/final_sparse_autoencoder_othello-gpt_blocks.6.hook_resid_pre_1024.pt"

model, sparse_autoencoder, activations_loader = (
    LMSparseAutoencoderSessionloader.load_session_from_pretrained(path)
)
sparse_autoencoder.eval()
# %%
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

# %%
import sys
import numpy as np
import torch
torch.set_grad_enabled(False)

from pathlib import Path

!git clone https://github.com/likenneth/othello_world
OTHELLO_ROOT = Path("./othello_world/")
sys.path.append(str(OTHELLO_ROOT/"mechanistic_interpretability"))

from mech_interp_othello_utils import plot_single_board, to_string, to_int, int_to_label, string_to_label, OthelloBoardState

board_seqs_int = torch.tensor(np.load(OTHELLO_ROOT/"mechanistic_interpretability/board_seqs_int_small.npy"), dtype=torch.long)
board_seqs_string = torch.tensor(np.load(OTHELLO_ROOT/"mechanistic_interpretability/board_seqs_string_small.npy"), dtype=torch.long)

num_games, length_of_game = board_seqs_int.shape
print("Number of games:", num_games,)
print("Length of game:", length_of_game)

stoi_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]
alpha = "ABCDEFGH"

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

board_labels = list(map(to_board_label, stoi_indices))

moves_int = board_seqs_int[0, :30]

# This is implicitly converted to a batch of size 1
logits = model(moves_int)
print("logits:", logits.shape)
# %%
# Test run
print(moves_int)
# %%
logit_vec = logits[0, -1]
log_probs = logit_vec.log_softmax(-1)
# Remove passing
log_probs = log_probs[1:]
assert len(log_probs)==60

temp_board_state = torch.zeros(64, device=logit_vec.device)
# Set all cells to -15 by default, for a very negative log prob - this means the middle cells don't show up as mattering
temp_board_state -= 13.
temp_board_state[stoi_indices] = log_probs

def plot_square_as_board(state, diverging_scale=True, **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    if diverging_scale:
        imshow(state, y=[i for i in alpha], x=[str(i) for i in range(8)], color_continuous_scale="RdBu", color_continuous_midpoint=0., aspect="equal", **kwargs)
    else:
        imshow(state, y=[i for i in alpha], x=[str(i) for i in range(8)], color_continuous_scale="Blues", color_continuous_midpoint=None, aspect="equal", **kwargs)
plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title="Example Log Probs")
# %%
plot_single_board(int_to_label(moves_int))
# %%
logits, cache = model.run_with_cache(moves_int)
cache: ActivationCache
# %%
imshow(cache.stack_activation("resid_pre").squeeze().norm(dim=-1))

resid_stack, labels = cache.get_full_resid_decomposition(expand_neurons=True, return_labels=True, pos_slice=0)
resid_stack = resid_stack.squeeze()
line(resid_stack.norm(dim=-1), x=labels)
# %%
ni = 1761
win = model.W_in[0, :, 1761]
temp_stack, temp_labels = cache.get_full_resid_decomposition(0, mlp_input=True, return_labels=True, pos_slice=0)
temp_stack = temp_stack.squeeze()
line(temp_stack @ win, x=temp_labels)
# %%
line(model.W_pos[0] @ model.W_in)
# %%
starting_moves = torch.tensor([[to_int("C3")], [to_int("D2")], [to_int("E5")], [to_int("F4")], ])
starting_logits, starting_cache = model.run_with_cache(starting_moves)

# %%
line(starting_cache["post", 0][:, 0, :])
# %%
imshow(cache.stack_activation("resid_pre").squeeze().norm(dim=-1)[:, 1:])
# %%
logits, cache = model.run_with_cache(board_seqs_int[:20, :-1])
imshow(cache.stack_activation("resid_post").norm(dim=-1).mean(dim=1)[:, 1:])
# %%
dataset_path = 'taufeeque/othellogpt'
model_name = 'othello-gpt'
device = "cuda" if torch.cuda.is_available() else "cpu"
# from sae_training.config import LanguageModelSAERunnerConfig
# exp_factor=1
# new_sae_config = LanguageModelSAERunnerConfig(
#         model_name=model_name,
#         hook_point="blocks.6.hook_resid_pre",
#         hook_point_layer=6,
#         dataset_path=dataset_path,
#         context_size=59,
#         d_in=512,
#         n_batches_in_buffer=32,
#         total_training_tokens=100*(1e6), # prev: 10*(1e6)
#         store_batch_size=32,
#         device=device,
#         seed=42,
#         dtype=torch.float32,
#         b_dec_init_method="geometric_median", # todo: geometric_median
#         expansion_factor=exp_factor, # todo: adjust
#         l1_coefficient=0.0002, # prev: 0.001, 0.0001
#         lr=0.00003, # prev: 0.0003
#         lr_scheduler_name="constantwithwarmup",
#         lr_warm_up_steps=5000,
#         train_batch_size=4096,
#         use_ghost_grads=True,
#         feature_sampling_window=500,
#         dead_feature_window=1e6,
#         log_to_wandb=False,
#         wandb_project="othello_gpt_sae",
#         wandb_log_frequency=30,
#         n_checkpoints=0,
#         checkpoint_path="checkpoints",
#         start_pos_offset=5, # exclude first seq position
#         end_pos_offset=-5
#     )
# %%
# sparse_autoencoder.cfg = new_sae_config
# %%
from sae_training.activations_store import ActivationsStore
# act_store = ActivationsStore(new_sae_config, model)
# %%
# act_store.get_activations(board_seqs_int[:20, :-1]).shape
# %%
import sae_training.evals
# sparse_autoencoder.cfg.start_pos_offset = 5
# sparse_autoencoder.cfg.end_pos_offset = -5
# sae_training.evals.get_recons_loss(sparse_autoencoder, model, act_store, board_seqs_int[:20, :-1])
# %%
full_linear_probe = torch.load(OTHELLO_ROOT/"mechanistic_interpretability/main_linear_probe.pth")

rows = 8
cols = 8 
options = 3
black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2
linear_probe = torch.zeros(model.cfg.d_model, rows, cols, options, device="cuda")
linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] + full_linear_probe[white_to_play_index, ..., 0])
linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] + full_linear_probe[white_to_play_index, ..., 2])
linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] + full_linear_probe[white_to_play_index, ..., 1])
# %%
linear_probe.shape
# %%
blank_probe = linear_probe[..., 0] - (linear_probe[..., 1] + linear_probe[..., 2])/2
blank_probe = blank_probe / blank_probe.norm(dim=0, keepdim=True)
their_probe = linear_probe[..., 1] - linear_probe[..., 2]
their_probe = their_probe / their_probe.norm(dim=0, keepdim=True)

print(f"{blank_probe.shape=}")
print(f"{their_probe.shape=}")
# %%

W_dec_normed = sparse_autoencoder.W_dec / sparse_autoencoder.W_dec.norm(dim=-1, keepdim=True)
W_enc_normed = sparse_autoencoder.W_enc.T / sparse_autoencoder.W_enc.T.norm(dim=-1, keepdim=True)
print(f"{W_dec_normed.shape=}")
print(f"{W_enc_normed.shape=}")
line((W_dec_normed @ W_enc_normed.T).diag())
# %%
all_board_labels = [string_to_label(i) for i in range(64)]
line((W_dec_normed @ blank_probe.reshape(d_model, 64)).T, line_labels=all_board_labels, title="Dec vs blank")
line((W_enc_normed @ blank_probe.reshape(d_model, 64)).T, line_labels=all_board_labels, title="Enc vs blank")
line((W_dec_normed @ their_probe.reshape(d_model, 64)).T, line_labels=all_board_labels, title="Dec vs their")
line((W_enc_normed @ their_probe.reshape(d_model, 64)).T, line_labels=all_board_labels, title="Enc vs their")
# %%
dec_board_sims = (W_dec_normed @ blank_probe.reshape(d_model, 64)).T
# %%
W_dec_probe_sims = W_dec_normed @ torch.cat([blank_probe.reshape(d_model, 64), their_probe.reshape(d_model, 64)], dim=1)
double_board_labels = [i+suffix for suffix in ["_blank", "_theirs"] for i in all_board_labels]
imshow(W_dec_normed @ torch.cat([blank_probe.reshape(d_model, 64), their_probe.reshape(d_model, 64)], dim=1), yaxis="Latent", xaxis="Board", x=double_board_labels, title="SAE Latents vs Probes")
line(W_dec_probe_sims.std(0), x=double_board_labels, title="SAE Latents vs Probes")
# %%
all_probe = torch.cat([blank_probe.reshape(d_model, 64), their_probe.reshape(d_model, 64)], dim=1)

W_dec = sparse_autoencoder.W_dec

W_dec_normed = W_dec / W_dec.norm(dim=-1, keepdim=True)
W_dec_probe_sims_abs = (W_dec_normed @ all_probe).abs()

max_sim_per_latent = W_dec_probe_sims_abs.max(dim=1).values
line(max_sim_per_latent, title="max_sim_per_latent")
max_sim_per_square = W_dec_probe_sims_abs.max(dim=0).values
line(max_sim_per_square, title="max_sim_per_square", x=double_board_labels)
# %%
rand_W_dec = torch.randn_like(sparse_autoencoder.W_dec)

rand_W_dec_normed = rand_W_dec / rand_W_dec.norm(dim=-1, keepdim=True)
rand_W_dec_probe_sims_abs = (rand_W_dec_normed @ all_probe).abs()

rand_max_sim_per_latent = rand_W_dec_probe_sims_abs.max(dim=1).values
line([rand_max_sim_per_latent, max_sim_per_latent], title="rand_max_sim_per_latent", line_labels=["rand", "original"])
rand_max_sim_per_square = rand_W_dec_probe_sims_abs.max(dim=0).values
line([rand_max_sim_per_square, max_sim_per_square], title="rand_max_sim_per_square", line_labels=["rand", "original"], x=double_board_labels)

# %%
num_games = int(1e3)
focus_games_int = board_seqs_int[:num_games, :-5]
focus_games_string = board_seqs_string[:num_games]
focus_logits, focus_cache = model.run_with_cache(focus_games_int, names_filter=utils.get_act_name("resid_pre", 6))
focus_resids = focus_cache["resid_pre", 6]
focus_resids_recons, *_, focus_sae_latents = sparse_autoencoder(focus_resids, return_pre=True)
# %%
def one_hot(list_of_ints, num_classes=64):
    out = torch.zeros((num_classes,), dtype=torch.float32)
    out[list_of_ints] = 1.
    return out
focus_states = np.zeros((num_games, 55, 8, 8), dtype=np.float32)
focus_valid_moves = torch.zeros((num_games, 55, 64), dtype=torch.float32)
for i in (range(num_games)):
    board = OthelloBoardState()
    for j in range(55):
        board.umpire(focus_games_string[i, j].item())
        focus_states[i, j] = board.state
        focus_valid_moves[i, j] = one_hot(board.get_valid_moves())
print("focus states:", focus_states.shape)
print("focus_valid_moves", focus_valid_moves.shape)

def state_stack_to_one_hot(state_stack):
    one_hot = torch.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        8, # rows
        8, # cols
        3, # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[..., 0] = state_stack == 0 # empty
    one_hot[..., 1] = state_stack == -1 # white
    one_hot[..., 2] = state_stack == 1 # black
    
    return one_hot

# We first convert the board states to be in terms of my (+1) and their (-1)
alternating = np.array([-1 if i%2 == 0 else 1 for i in range(focus_games_int.shape[1])])
flipped_focus_states = focus_states * alternating[None, :, None, None]

# We now convert to one hot
focus_states_flipped_one_hot = state_stack_to_one_hot(torch.tensor(flipped_focus_states))

# Take the argmax
focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
# %%
probe_out = einops.einsum(focus_cache["resid_pre", 6], linear_probe, "game move d_model, d_model row col options -> game move row col options")
probe_out_value = probe_out.argmax(dim=-1)

# correct_middle_odd_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :])[:, 5:-5:2]
# accuracies_odd = einops.reduce(correct_middle_odd_answers.float(), "game move row col -> row col", "mean")
correct_middle_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :])[:, ]
accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col", "mean")

plot_square_as_board(1 - accuracies, title="Average Error Rate of Linear Probe", zmax=0.25, zmin=-0.25)
# %%
l_id = 49
latent_acts = F.relu(focus_sae_latents[:, 5:, l_id].flatten())
is_firing = latent_acts > 0
reduced_states_one_hot = einops.rearrange(focus_states_flipped_one_hot[:, 5:, :, :, :], "game move row col opts -> (game move) (row col) opts").cuda()
print(reduced_states_one_hot[0, 0])
out_acts = torch.zeros((64, 3)).cuda()
out_frac = torch.zeros((64, 3)).cuda()
for i in range(64):
    for j in range(3):
        out_acts[i, j] = latent_acts[reduced_states_one_hot[:, i, j]>0].mean()
        out_frac[i, j] = is_firing[reduced_states_one_hot[:, i, j]>0].float().mean()
line(out_frac.T, line_labels=["blank", "their's", "mine"], title="Fraction of time firing", x=all_board_labels)
line(out_acts.T, line_labels=["blank", "their's", "mine"], title="Average Firing", x=all_board_labels)
# %%
move_int = to_int("C5")
move_string = to_string("C5")
move_index = 21
was_empty = reduced_states_one_hot[:, move_index, 0]>0
px.histogram(to_numpy(focus_sae_latents[:, 5:, l_id].flatten()), color=to_numpy(was_empty), barmode="overlay")
# %%
line((sparse_autoencoder.W_dec[l_id] / sparse_autoencoder.W_dec[l_id].norm() ) @ (model.W_U / model.W_U.norm(dim=0, keepdim=True)))
line((sparse_autoencoder.W_enc[:, l_id] / sparse_autoencoder.W_enc[:, l_id].norm() ) @ (model.W_U / model.W_U.norm(dim=0, keepdim=True)))
line((blank_probe[:, 2, 5] / blank_probe[:, 2, 5].norm() ) @ (model.W_U / model.W_U.norm(dim=0, keepdim=True)))
# %%
c5_log_probs = focus_logits.log_softmax(dim=-1)[:, 5:, 22].flatten()
histogram(c5_log_probs)
# %%
scatter(focus_sae_latents[:, 5:, l_id].flatten(), c5_log_probs, opacity=0.1, color=was_empty)
# %%
from sklearn.model_selection import train_test_split
y = to_numpy(c5_log_probs > -6)
X = to_numpy(focus_resids[:, 5:, :].reshape(50000, d_model))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
new_probe = LogisticRegression()
new_probe.fit(X_train, y_train)
(new_probe.predict(X_test)==y_test).astype(np.float32).mean()
# %%
y = to_numpy(c5_log_probs > -6)
X = to_numpy(focus_resids[:, 5:, :].reshape(50000, d_model) @ sparse_autoencoder.W_enc[:, l_id])[:, None]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
new_probe = LogisticRegression()
new_probe.fit(X_train, y_train)
(new_probe.predict(X_test)==y_test).astype(np.float32).mean()
# %%
labels = c5_log_probs<-4
temp_tensor = torch.clone(focus_sae_latents[:, 5:, l_id].flatten())
temp_tensor[~labels] = torch.inf
temp_tensor = -temp_tensor
indices = temp_tensor.topk(100).indices
games = indices // 50
moves = (indices % 50) + 5
print(games)
print(moves)
# %%
states = []
for i in range(100):
    states.append(focus_states[games[i], moves[i]])
    # plot_single_board(focus_games_string[games[i], :moves[i]+1])
# imshow(np.stack(states), facet_col=0)
imshow(np.stack(states).mean(0), title="Ave")
imshow(np.abs(np.stack(states)).mean(0), title="Ave abs")
# %%
px.scatter(x=to_numpy(focus_sae_latents[:, 5:, l_id].flatten()[was_empty]), y=to_numpy(c5_log_probs[was_empty]), color=to_numpy(c5_log_probs[was_empty]>-5), opacity=0.1, marginal_x="histogram", marginal_y="histogram")
# %%
var_names = []
for l in "ABCDEFGH":
    for i in range(8):
        for suffix in ["blank", "mine", "their's"]:
            var_names.append(f"{l}{i}_{suffix}")
variables = einops.rearrange(focus_states_flipped_one_hot[:, 5:, :, :, :], "game move row col opts -> (game move) (row col opts)").cuda()
print(variables.shape)

is_valid_by_move = focus_logits.log_softmax(dim=-1)[:, 5:, 1:].reshape(50000, 60) > -5
variables = torch.cat([variables, is_valid_by_move], dim=-1)
for i in range(1, 61):
    var_names.append(f"is_{int_to_label(i)}_valid")
# %%
variables_num_true = variables.sum(dim=0)
variables_num_false = len(variables) - variables_num_true
line(variables_num_false, x=var_names)
# %%
exp_factor = 2
d_sae = d_model * exp_factor
focus_sae_latents_flat = F.relu(focus_sae_latents[:, 5:].reshape(50000, d_sae))
ave_latent_if_true = (focus_sae_latents_flat.T @ variables.float()) / (variables_num_true[None, :]+1e-6)
ave_latent_if_false = (focus_sae_latents_flat.T @ (1-variables.float())) / variables_num_false[None, :]
ave_latent_diff = ave_latent_if_true - ave_latent_if_false

line(ave_latent_diff[l_id], x=var_names)
# %%
frac_latent_if_true = ((focus_sae_latents_flat>0).float().T @ variables.float()) / (variables_num_true[None, :]+1e-6)
frac_latent_if_false = ((focus_sae_latents_flat>0).float().T @ (1-variables.float())) / variables_num_false[None, :]
frac_latent_diff = frac_latent_if_true - frac_latent_if_false
line(frac_latent_diff[l_id], x=var_names)

# %%
line([frac_latent_diff.max(-1).values, ave_latent_diff.max(-1).values], line_labels=["frac", "ave"])
# %%
focus_sae_latents_flat_no_relu = (focus_sae_latents[:, 5:].reshape(50000, d_sae))
ave_latent_no_relu_if_true = (focus_sae_latents_flat_no_relu.T @ variables.float()) / (variables_num_true[None, :]+1e-6)
ave_latent_no_relu_if_false = (focus_sae_latents_flat_no_relu.T @ (1-variables.float())) / variables_num_false[None, :]
ave_latent_no_relu_diff = ave_latent_no_relu_if_true - ave_latent_no_relu_if_false

line(ave_latent_no_relu_diff[l_id], x=var_names)
# %%
line([frac_latent_diff[110], ave_latent_diff[110]], x=var_names, line_labels=["frac", "ave"])
# %%
var_id = var_names.index("is_E5_valid")
line(frac_latent_diff[:, var_id])
# %%
px.histogram(to_numpy(focus_sae_latents_flat_no_relu[:, 110]), color=to_numpy(variables[:, var_id]), marginal="box", title="Latent 110 Detects If E5 Is Valid", labels={"x": "Latent 110", "color": "Is E5 Valid"}, histnorm="percent")
# %%
ave_latent_diff.max(dim=-1)
# %%
max_frac_per_latent = frac_latent_diff.max(dim=-1).values
max_frac_per_latent_no_valid = frac_latent_diff[:, :-60].max(dim=-1).values
scatter(x=max_frac_per_latent, y=max_frac_per_latent_no_valid, hover=np.arange(d_sae), xaxis="Max Frac Diff Firing Per Latent", yaxis="Max Frac Diff Firing Per Latent (No Valid)", title="Max Frac Diff Firing Per Latent vs Max Frac Diff Firing Per Latent (No Valid)")

# %%
def plot_latent(l_id):
    line([frac_latent_diff[l_id], ave_latent_diff[l_id]], x=var_names, line_labels=["frac", "ave"], title=f"Latent {l_id} Diff Firing")
plot_latent(394)
plot_latent(539)
plot_latent(805)
plot_latent(383)
# %%
subvars = []
sub_var_names = []
for i in range(len(var_names)):
    if ("mine" in var_names[i] or "their" in var_names[i]) and var_names[i][:2] not in ["D3", "D4", "E3", "E4"]:
        subvars.append(i)
        sub_var_names.append(var_names[i])
subvars = np.array(subvars)
sub_variables = variables[:, subvars]
print(subvars)
print(sub_variables.shape, len(subvars))
# %%
mine_variables = sub_variables[:, ::2]
their_variables = sub_variables[:, 1::2]
board_labels_excl_center = [i for i in all_board_labels if i not in ["D3", "D4", "E3", "E4"]]
num_mine = mine_variables.sum(dim=0)
num_their = their_variables.sum(dim=0)
line([num_mine, num_their], x=board_labels_excl_center)
# %%
frac_latent_cond_mine = ((focus_sae_latents_flat>0).float().T @ mine_variables.float()) / (num_mine[None, :])
frac_latent_cond_their = ((focus_sae_latents_flat>0).float().T @ (their_variables.float())) / num_their[None, :]
frac_latent_cond_diff = frac_latent_cond_mine - frac_latent_cond_their
line(frac_latent_cond_diff.abs().max(dim=-1).values)
# %%
line(frac_latent_cond_diff[208], x=board_labels_excl_center)
# %%
l_id = 208
temp_tensor = torch.clone(focus_sae_latents[:, 5:, l_id].flatten())
# temp_tensor[~labels] = torch.inf
# temp_tensor = -temp_tensor
indices = temp_tensor.topk(500).indices
games = indices // 50
moves = (indices % 50) + 5
print(games)
print(moves)

states = []
for i in range(500):
    states.append(focus_states[games[i], moves[i]])
    # plot_single_board(focus_games_string[games[i], :moves[i]+1])
# imshow(np.stack(states), facet_col=0)
states = np.stack(states)
imshow((states).mean(0), title="Ave")
imshow(np.abs((states)).mean(0), title="Ave abs")

for r in [0, 1]:
    for c in [6, 7]:
        print("ABCDEFGH"[r], c)
        print(pd.Series(states[:, r, c]).value_counts())
        print()
line(states[:, [0, 0, 1, 1, 0, 2], [6, 7, 6, 7, 5, 5]].reshape(500, -1).T, line_labels=["A6", "A7", "B6", "B7", "A5", "C5"])

imshow((states == states[:, 0, 7][:, None, None]).mean(0), title="Ave Agrees with A7")
# %%
l_id = 208
w_enc = sparse_autoencoder.W_enc[:, l_id]
w_enc = w_enc / w_enc.norm()
W_in = torch.randn_like(model.W_in[:6])
W_in = W_in / W_in.norm(dim=-2, keepdim=True)
line(w_enc @ W_in)
# %%
proxy = ((focus_states_flipped_value[:, 5:, 0, 7]!=0) & (focus_states_flipped_value[:, 5:, 0, 7]==focus_states_flipped_value[:, 5:, 0, 6]) & (focus_states_flipped_value[:, 5:, 0, 7]==focus_states_flipped_value[:, 5:, 1, 6])).flatten()
# alt_proxy = ((focus_states_flipped_value[:, 5:, 0, 7]==focus_states_flipped_value[:, 5:, 0, 5]) & (focus_states_flipped_value[:, 5:, 0, 7]!=0)).flatten()
alt_proxy = focus_states_flipped_value[:, 5:, 0, 7].flatten()
color = proxy * 3 + alt_proxy
# color = proxy
px.histogram(to_numpy(focus_sae_latents_flat_no_relu[:, l_id]), color=to_numpy(color), barmode="overlay", histnorm="percent", marginal="box")
# %%
r = 3
c = 2
non_empty = torch.ones_like(focus_states_flipped_value[:, 5:, r, c] != 0).flatten()
sae_latents = F.relu(focus_sae_latents[:, 5:].reshape(50000, d_sae)[non_empty])
is_mine = (focus_states_flipped_value[:, 5:, r, c] == 0).flatten()[non_empty]
ave_diff = sae_latents[is_mine].mean(0) - sae_latents[~is_mine].mean(0)

accs = []
ks = [1, 2, 5, 10, 20, 50, 100, 1024]
for k in ks:
    # if make_rand:
    #     indices = (-torch.randn_like(ave_diff.abs())).argsort()
    # else:
    indices = (-(ave_diff.abs())).argsort()
    X = to_numpy(sae_latents[:, indices[:k]])
    y = to_numpy(is_mine)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    new_probe = LogisticRegression()
    new_probe.fit(X_train, y_train)
    acc = (new_probe.predict(X_test)==y_test).astype(np.float32).mean()
    print("Accuracy", acc)
    accs.append(float(acc))
line(x=ks, y=accs, title="Accuracy vs K", xaxis="K", yaxis="Accuracy")
# %%
focus_resids_recons.shape
# %%
recons_acc = (einops.einsum(focus_resids_recons[:, 5:, :].reshape(50000, d_model), linear_probe, "token d_model, d_model row col opt -> token row col opt").argmax(dim=-1)==(focus_states_flipped_value[:, 5:].reshape(50000, 8, 8).cuda())).float().mean(0)
resid_acc = (einops.einsum(focus_resids[:, 5:, :].reshape(50000, d_model), linear_probe, "token d_model, d_model row col opt -> token row col opt").argmax(dim=-1)==(focus_states_flipped_value[:, 5:].reshape(50000, 8, 8).cuda())).float().mean(0)
imshow([recons_acc, resid_acc], facet_col=0, facet_labels=["recons", "orig"])
imshow(resid_acc - recons_acc)
# imshow((einops.einsum(focus_resids[:, 5:, :].reshape(50000, d_model), linear_probe, "token d_model, d_model row col opt -> token row col opt").argmax(dim=-1)==(focus_states_flipped_value[:, 5:].reshape(50000, 8, 8).cuda())).float().mean(0))
# %%
