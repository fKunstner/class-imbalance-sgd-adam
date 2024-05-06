from dataclasses import dataclass

from optexp.problems import Classification, FullBatchClassification


@dataclass
class Transformer(Classification):
    pass


class FullBatchTransformer(FullBatchClassification):
    pass
