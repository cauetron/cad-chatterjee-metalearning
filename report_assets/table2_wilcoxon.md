# Table 2 — Paired Wilcoxon signed-rank tests (α = 0.05)

| Scope   | Meta-learner   | Metric   | Comparison       |   N pairs |   Mean baseline |   Mean proposal |   Mean Δ |   Median Δ | p-value   | Sig. (α=0.05)   | Winner   |
|:--------|:---------------|:---------|:-----------------|----------:|----------------:|----------------:|---------:|-----------:|:----------|:----------------|:---------|
| run     | kNN            | SRC      | CaD-Sp vs CaD-Ch |      1240 |          0.6114 |          0.6437 |   0.0323 |     0      | <0.0001   | ✓               | CaD-Ch   |
| run     | kNN            | SRC      | CaD-Sp vs CaD-Ke |      1240 |          0.6114 |          0.6079 |  -0.0034 |     0      | 0.6358    |                 | tie      |
| run     | kNN            | WRC      | CaD-Sp vs CaD-Ch |      1240 |          0.5487 |          0.5858 |   0.037  |     0      | <0.0001   | ✓               | CaD-Ch   |
| run     | kNN            | WRC      | CaD-Sp vs CaD-Ke |      1240 |          0.5487 |          0.5403 |  -0.0084 |     0      | 0.0075    | ✓               | CaD-Sp   |
| run     | kNN            | NDCG@3   | CaD-Sp vs CaD-Ch |      1240 |          0.8925 |          0.9036 |   0.0112 |     0      | <0.0001   | ✓               | CaD-Ch   |
| run     | kNN            | NDCG@3   | CaD-Sp vs CaD-Ke |      1240 |          0.8925 |          0.8903 |  -0.0022 |     0      | 0.2879    |                 | tie      |
| run     | kNN            | Top1Hit  | CaD-Sp vs CaD-Ch |      1240 |          0.5734 |          0.6274 |   0.054  |     0      | 0.0014    | ✓               | CaD-Ch   |
| run     | kNN            | Top1Hit  | CaD-Sp vs CaD-Ke |      1240 |          0.5734 |          0.5185 |  -0.0548 |     0      | 0.0009    | ✓               | CaD-Sp   |
| run     | kNN            | Regret@1 | CaD-Sp vs CaD-Ch |      1240 |          0.1971 |          0.1696 |  -0.0275 |     0      | <0.0001   | ✓               | CaD-Ch   |
| run     | kNN            | Regret@1 | CaD-Sp vs CaD-Ke |      1240 |          0.1971 |          0.2181 |   0.0211 |     0      | 0.0006    | ✓               | CaD-Sp   |
| run     | kNN            | ARI      | CaD-Sp vs CaD-Ch |      1240 |          0.1644 |          0.164  |  -0.0004 |     0      | 0.4811    |                 | tie      |
| run     | kNN            | ARI      | CaD-Sp vs CaD-Ke |      1240 |          0.1644 |          0.1699 |   0.0056 |     0      | 0.0128    | ✓               | CaD-Ke   |
| run     | kNN            | AMI      | CaD-Sp vs CaD-Ch |      1240 |          0.2269 |          0.225  |  -0.0019 |     0      | 0.0193    | ✓               | CaD-Sp   |
| run     | kNN            | AMI      | CaD-Sp vs CaD-Ke |      1240 |          0.2269 |          0.2281 |   0.0012 |     0      | 0.2391    |                 | tie      |
| run     | RF             | SRC      | CaD-Sp vs CaD-Ch |      1240 |          0.6462 |          0.6441 |  -0.0021 |     0      | 0.0343    | ✓               | CaD-Sp   |
| run     | RF             | SRC      | CaD-Sp vs CaD-Ke |      1240 |          0.6462 |          0.6253 |  -0.0209 |     0      | <0.0001   | ✓               | CaD-Sp   |
| run     | RF             | WRC      | CaD-Sp vs CaD-Ch |      1240 |          0.5833 |          0.5868 |   0.0035 |     0      | 0.4854    |                 | tie      |
| run     | RF             | WRC      | CaD-Sp vs CaD-Ke |      1240 |          0.5833 |          0.5696 |  -0.0137 |     0      | 0.0011    | ✓               | CaD-Sp   |
| run     | RF             | NDCG@3   | CaD-Sp vs CaD-Ch |      1240 |          0.9043 |          0.9036 |  -0.0007 |     0      | 0.0554    |                 | tie      |
| run     | RF             | NDCG@3   | CaD-Sp vs CaD-Ke |      1240 |          0.9043 |          0.9    |  -0.0043 |     0      | 0.0141    | ✓               | CaD-Sp   |
| run     | RF             | Top1Hit  | CaD-Sp vs CaD-Ch |      1240 |          0.5935 |          0.6121 |   0.0185 |     0      | 0.2848    |                 | tie      |
| run     | RF             | Top1Hit  | CaD-Sp vs CaD-Ke |      1240 |          0.5935 |          0.5798 |  -0.0137 |     0      | 0.3822    |                 | tie      |
| run     | RF             | Regret@1 | CaD-Sp vs CaD-Ch |      1240 |          0.1716 |          0.1737 |   0.0021 |     0      | 0.7766    |                 | tie      |
| run     | RF             | Regret@1 | CaD-Sp vs CaD-Ke |      1240 |          0.1716 |          0.1947 |   0.0231 |     0      | 0.0624    |                 | tie      |
| run     | RF             | ARI      | CaD-Sp vs CaD-Ch |      1240 |          0.1732 |          0.1691 |  -0.0042 |     0      | <0.0001   | ✓               | CaD-Sp   |
| run     | RF             | ARI      | CaD-Sp vs CaD-Ke |      1240 |          0.1732 |          0.1726 |  -0.0006 |     0      | 0.1068    |                 | tie      |
| run     | RF             | AMI      | CaD-Sp vs CaD-Ch |      1240 |          0.231  |          0.2309 |  -0.0002 |     0      | 0.0817    |                 | tie      |
| run     | RF             | AMI      | CaD-Sp vs CaD-Ke |      1240 |          0.231  |          0.2302 |  -0.0008 |     0      | 0.8075    |                 | tie      |
| dataset | kNN            | SRC      | CaD-Sp vs CaD-Ch |        62 |          0.6114 |          0.6437 |   0.0323 |     0.0211 | 0.2962    |                 | tie      |
| dataset | kNN            | SRC      | CaD-Sp vs CaD-Ke |        62 |          0.6114 |          0.6079 |  -0.0034 |     0      | 0.9692    |                 | tie      |
| dataset | kNN            | WRC      | CaD-Sp vs CaD-Ch |        62 |          0.5487 |          0.5858 |   0.037  |     0.0156 | 0.2680    |                 | tie      |
| dataset | kNN            | WRC      | CaD-Sp vs CaD-Ke |        62 |          0.5487 |          0.5403 |  -0.0084 |    -0.0083 | 0.2517    |                 | tie      |
| dataset | kNN            | NDCG@3   | CaD-Sp vs CaD-Ch |        62 |          0.8925 |          0.9036 |   0.0112 |     0.0049 | 0.0578    |                 | tie      |
| dataset | kNN            | NDCG@3   | CaD-Sp vs CaD-Ke |        62 |          0.8925 |          0.8903 |  -0.0022 |     0      | 0.9329    |                 | tie      |
| dataset | kNN            | Top1Hit  | CaD-Sp vs CaD-Ch |        62 |          0.5734 |          0.6274 |   0.054  |     0      | 0.3137    |                 | tie      |
| dataset | kNN            | Top1Hit  | CaD-Sp vs CaD-Ke |        62 |          0.5734 |          0.5185 |  -0.0548 |     0      | 0.4162    |                 | tie      |
| dataset | kNN            | Regret@1 | CaD-Sp vs CaD-Ch |        62 |          0.1971 |          0.1696 |  -0.0275 |     0      | 0.3472    |                 | tie      |
| dataset | kNN            | Regret@1 | CaD-Sp vs CaD-Ke |        62 |          0.1971 |          0.2181 |   0.0211 |     0      | 0.4613    |                 | tie      |
| dataset | kNN            | ARI      | CaD-Sp vs CaD-Ch |        62 |          0.1644 |          0.164  |  -0.0004 |     0      | 0.4683    |                 | tie      |
| dataset | kNN            | ARI      | CaD-Sp vs CaD-Ke |        62 |          0.1644 |          0.1699 |   0.0056 |     0      | 0.6764    |                 | tie      |
| dataset | kNN            | AMI      | CaD-Sp vs CaD-Ch |        62 |          0.2269 |          0.225  |  -0.0019 |     0      | 0.2767    |                 | tie      |
| dataset | kNN            | AMI      | CaD-Sp vs CaD-Ke |        62 |          0.2269 |          0.2281 |   0.0012 |     0      | 0.7448    |                 | tie      |
| dataset | RF             | SRC      | CaD-Sp vs CaD-Ch |        62 |          0.6462 |          0.6441 |  -0.0021 |    -0.0222 | 0.3511    |                 | tie      |
| dataset | RF             | SRC      | CaD-Sp vs CaD-Ke |        62 |          0.6462 |          0.6253 |  -0.0209 |    -0.0098 | 0.0958    |                 | tie      |
| dataset | RF             | WRC      | CaD-Sp vs CaD-Ch |        62 |          0.5833 |          0.5868 |   0.0035 |    -0.0135 | 0.6689    |                 | tie      |
| dataset | RF             | WRC      | CaD-Sp vs CaD-Ke |        62 |          0.5833 |          0.5696 |  -0.0137 |    -0.0054 | 0.2679    |                 | tie      |
| dataset | RF             | NDCG@3   | CaD-Sp vs CaD-Ch |        62 |          0.9043 |          0.9036 |  -0.0007 |    -0.0022 | 0.4040    |                 | tie      |
| dataset | RF             | NDCG@3   | CaD-Sp vs CaD-Ke |        62 |          0.9043 |          0.9    |  -0.0043 |     0      | 0.5859    |                 | tie      |
| dataset | RF             | Top1Hit  | CaD-Sp vs CaD-Ch |        62 |          0.5935 |          0.6121 |   0.0185 |     0      | 0.8953    |                 | tie      |
| dataset | RF             | Top1Hit  | CaD-Sp vs CaD-Ke |        62 |          0.5935 |          0.5798 |  -0.0137 |     0      | 0.7867    |                 | tie      |
| dataset | RF             | Regret@1 | CaD-Sp vs CaD-Ch |        62 |          0.1716 |          0.1737 |   0.0021 |     0      | 0.9604    |                 | tie      |
| dataset | RF             | Regret@1 | CaD-Sp vs CaD-Ke |        62 |          0.1716 |          0.1947 |   0.0231 |     0      | 0.7215    |                 | tie      |
| dataset | RF             | ARI      | CaD-Sp vs CaD-Ch |        62 |          0.1732 |          0.1691 |  -0.0042 |     0      | 0.3123    |                 | tie      |
| dataset | RF             | ARI      | CaD-Sp vs CaD-Ke |        62 |          0.1732 |          0.1726 |  -0.0006 |     0      | 0.8042    |                 | tie      |
| dataset | RF             | AMI      | CaD-Sp vs CaD-Ch |        62 |          0.231  |          0.2309 |  -0.0002 |     0      | 0.8788    |                 | tie      |
| dataset | RF             | AMI      | CaD-Sp vs CaD-Ke |        62 |          0.231  |          0.2302 |  -0.0008 |     0      | 0.6024    |                 | tie      |
