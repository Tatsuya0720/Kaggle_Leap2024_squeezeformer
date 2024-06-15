import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def graph_plot(epoch, r2_score_dict, weight):
    r2_score_df = pd.DataFrame(r2_score_dict.values(), index=r2_score_dict.keys()).T
    columns = r2_score_df.columns
    ptend_t_index = [i for i, c in enumerate(columns) if "ptend_t" in c]
    ptend_q0001_index = [i for i, c in enumerate(columns) if "ptend_q0001" in c]
    ptend_q0002_index = [i for i, c in enumerate(columns) if "ptend_q0002" in c]
    ptend_q0003_index = [i for i, c in enumerate(columns) if "ptend_q0003" in c]
    ptend_u_index = [i for i, c in enumerate(columns) if "ptend_u" in c]
    ptend_v_index = [i for i, c in enumerate(columns) if "ptend_v" in c]

    ptend_t_df = r2_score_df.iloc[0, ptend_t_index]
    ptend_q0001_df = r2_score_df.iloc[0, ptend_q0001_index]
    ptend_q0002_df = r2_score_df.iloc[0, ptend_q0002_index]
    ptend_q0003_df = r2_score_df.iloc[0, ptend_q0003_index]
    ptend_u_df = r2_score_df.iloc[0, ptend_u_index]
    ptend_v_df = r2_score_df.iloc[0, ptend_v_index]

    ptend_t_df_weight = np.where(weight[ptend_t_index] < 1, 0, 1)
    ptend_q0001_df_weight = np.where(weight[ptend_q0001_index] < 1, 0, 1)
    ptend_q0002_df_weight = np.where(weight[ptend_q0002_index] < 1, 0, 1)
    ptend_q0003_df_weight = np.where(weight[ptend_q0003_index] < 1, 0, 1)
    ptend_u_df_weight = np.where(weight[ptend_u_index] < 1, 0, 1)
    ptend_v_df_weight = np.where(weight[ptend_v_index] < 1, 0, 1)

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    # 縦軸を0から1に揃える
    for i in range(3):
        for j in range(2):
            ax[i, j].set_ylim(-1, 1)

    # 棒グラフではなく，丸でプロット
    ptend_t_df.plot(
        kind="bar", ax=ax[0, 0], color="skyblue", xticks=[], xlabel="ptend_t"
    )
    ax[0, 0].scatter(
        np.arange(len(ptend_t_df)),
        ptend_t_df_weight,
        color="red",
        marker="o",
        s=10,
    )
    ptend_q0001_df.plot(
        kind="bar", ax=ax[0, 1], color="skyblue", xticks=[], xlabel="ptend_q0001"
    )
    ax[0, 1].scatter(
        np.arange(len(ptend_q0001_df)),
        ptend_q0001_df_weight,
        color="red",
        marker="o",
        s=10,
    )
    ptend_q0002_df.plot(
        kind="bar", ax=ax[1, 0], color="skyblue", xticks=[], xlabel="ptend_q0002"
    )
    ax[1, 0].scatter(
        np.arange(len(ptend_q0002_df)),
        ptend_q0002_df_weight,
        color="red",
        marker="o",
        s=10,
    )
    ptend_q0003_df.plot(
        kind="bar", ax=ax[1, 1], color="skyblue", xticks=[], xlabel="ptend_q0003"
    )
    ax[1, 1].scatter(
        np.arange(len(ptend_q0003_df)),
        ptend_q0003_df_weight,
        color="red",
        marker="o",
        s=10,
    )
    ptend_u_df.plot(
        kind="bar", ax=ax[2, 0], color="skyblue", xticks=[], xlabel="ptend_u"
    )
    ax[2, 0].scatter(
        np.arange(len(ptend_u_df)), ptend_u_df_weight, color="red", marker="o", s=10
    )
    ptend_v_df.plot(
        kind="bar", ax=ax[2, 1], color="skyblue", xticks=[], xlabel="ptend_v"
    )
    ax[2, 1].scatter(
        np.arange(len(ptend_v_df)), ptend_v_df_weight, color="red", marker="o", s=10
    )
    # plotしたグラフを保存
    plt.savefig("./plots/epoch_{}.png".format(epoch))
    plt.show()
