## Results

|    | models        | r2        |
|---:|:--------------|:----------|
|  0 | CB_af         | 0.023     |
|  1 | CB_wout_emb_f | 0.009     |
|  2 | CB_tf         | 0.168     |
|  3 | CB_tf_fi      | 0.23      |
|  4 | CV_CB_bf      | -2.001    |
|  5 | ET_wout_emb_f | 0.357     |
|  6 | RF_wout_emb_f | 0.395     |
|  7 | stacking      | **0.397** |
|  8 | gru_condition | -1.188    |
|  9 | gru           | -60.457   |
| 10 | KAN           | -         |

`CB_af` - CatBoostRegressor with all features

`CB_wout_emb_f` - CatBoostRegressor without embedding features

`CB_tf` - CatBoostRegressor with target filtering

`CB_tf_fi` - CatBoostRegressor with target filtering + feature importance

`CV_CB_bf` - CrossValidation with CatBoostRegressor

`ET_wout_emb_f` - ExtraTreesRegressor without embedding features

`RF_wout_emb_f` - RandomForestRegressor without embedding features

`stacking` - Stacking: CatBoostRegressor+ExtraTreesRegressor+RandomForestRegressor

`KAN` - KAN with all features

`gru` - GRU only with a formula seq

`gru_condition` - GRU with 15 most important features and a formula seq 
