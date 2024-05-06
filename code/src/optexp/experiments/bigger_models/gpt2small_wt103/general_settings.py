from optexp import MixedBatchSizeTextDataset
from optexp.models.initializer import TransformerEncoderInitializer
from optexp.models.transformer_encoder import GPTModel

INIT_STD = 0.02
depth = 12
num_heads = 12
emb_dim = 768

avs_initializer = TransformerEncoderInitializer.default_scaled_by_depth(
    depth, (2**0.5) * INIT_STD
)

model = GPTModel(
    num_heads=num_heads,
    depth=depth,
    emb_dim=emb_dim,
    init=avs_initializer,
)


def get_dataset(train_batch_size, eval_batch_size, target_length):
    return MixedBatchSizeTextDataset(
        name="WikiText103",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        tgt_len=target_length,
        batch_size=None,  # type: ignore
    )


GROUP_5K = "GPT2Small_Wikitext103_Per_Class_Stats_5k_Steps"
GROUP_15K = "GPT2Small_Wikitext103_Per_Class_Stats_15k_Steps"
