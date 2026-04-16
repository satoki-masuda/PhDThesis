"""居住地選択モデルだけを単独で推定する簡易スクリプト。"""

from model.transition_model import TransitionModel

start_year = 2015
end_year = 2023
ref_year = 1
transition_model = TransitionModel(start_year=start_year, end_year=end_year, ref_year=ref_year, dev_zone="19_zone", res_zone="pop_zone")

X1_hetero, X1_mean, X2, Z, W, y, obs_share, relocation_years = transition_model.make_estimation_data_2step()
transition_model.estimate_choice_model(y, method="MNL_2step", X1_hetero=X1_hetero, X1_mean=X1_mean, X2=X2, Z=Z, W=W, obs_share=obs_share, relocation_years=relocation_years)
