import numpy as np
import pandas as pd
import pytest

from locpick import ChoiceTable


def _make_tables():
    choosers = pd.DataFrame(
        {
            "obs_feature": [1.0, 2.0, 3.0],
            "chosen_alt": [10, 11, 12],
        },
        index=pd.Index([1, 2, 3], name="obs_id"),
    )
    alternatives = pd.DataFrame(
        {
            "alt_feature": [0.5, 1.5, 2.5, 3.5],
        },
        index=pd.Index([10, 11, 12, 13], name="alt_id"),
    )
    return choosers, alternatives


def test_census_dataset_schema():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    assert ct._ds.sizes["obs_id"] == 3
    assert ct._ds.sizes["alt_pos"] == 4
    assert "alt_id_values" in ct._ds.data_vars
    assert ct._ds["alt_id_values"].dims == ("obs_id", "alt_pos")


def test_sampled_dataset_schema():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=42,
    )

    assert ct._ds.sizes["obs_id"] == 3
    assert ct._ds.sizes["alt_pos"] == 2
    assert ct._ds["alt_id_values"].shape == (3, 2)


def test_chosen_sum_matches_observations():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=3,
        replace=False,
        seed=1,
    )

    assert int(ct._ds["chosen"].values.sum()) == ct.n_observations


def test_from_tables_chosen_dataframe_census_allows_alt_id_column_name():
    choosers = pd.DataFrame({"id": [1, 2, 3], "obs_feature": [1.0, 2.0, 3.0]})
    alternatives = pd.DataFrame(
        {
            "alt_id": [10, 11, 12, 13],
            "alt_feature": [0.5, 1.5, 2.5, 3.5],
        }
    )
    chosen = pd.DataFrame({"id": [1, 2, 3], "alt_id": [10, 12, 13]})

    ct = ChoiceTable.from_tables(choosers, alternatives, chosen)
    frame = ct.to_frame()

    assert int(frame["chosen"].sum()) == 3
    chosen_rows = frame.loc[frame["chosen"] == 1, ["id", "alt_id"]].sort_values("id")
    expected = chosen.sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(chosen_rows.reset_index(drop=True), expected)


def test_from_tables_chosen_dataframe_sampled_allows_alt_id_column_name():
    choosers = pd.DataFrame({"id": [1, 2, 3], "obs_feature": [1.0, 2.0, 3.0]})
    alternatives = pd.DataFrame(
        {
            "alt_id": [10, 11, 12, 13],
            "alt_feature": [0.5, 1.5, 2.5, 3.5],
        }
    )
    chosen = pd.DataFrame({"id": [1, 2, 3], "alt_id": [10, 12, 13]})

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen,
        sample_size=3,
        replace=False,
        seed=44,
    )
    frame = ct.to_frame()

    assert int(frame["chosen"].sum()) == 3
    chosen_rows = frame.loc[frame["chosen"] == 1, ["id", "alt_id"]].sort_values("id")
    expected = chosen.sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(chosen_rows.reset_index(drop=True), expected)


def test_add_interaction_alignment_census_order_independent():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    idx = pd.MultiIndex.from_product(
        [choosers.index, alternatives.index], names=["obs_id", "alt_id"]
    )
    vals = pd.Series([obs * 100 + alt for obs, alt in idx], index=idx)
    vals = vals.sample(frac=1.0, random_state=99)

    ct2 = ct.add_interaction("distance", vals)
    frame = ct2.to_frame()
    expected = [vals.loc[(obs, alt)] for obs, alt in zip(frame["obs_id"], frame["alt_id"])]

    assert np.allclose(frame["distance"].to_numpy(dtype=float), np.asarray(expected, dtype=float))


def test_add_interaction_is_lazy_until_materialization():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    idx = pd.MultiIndex.from_product(
        [choosers.index, alternatives.index], names=["obs_id", "alt_id"]
    )
    vals = pd.Series([obs * 10 + alt for obs, alt in idx], index=idx)

    ct2 = ct.add_interaction("distance", vals)
    assert "distance" not in ct2._ds.data_vars

    frame = ct2.to_frame()
    expected = np.asarray(
        [vals.loc[(obs, alt)] for obs, alt in zip(frame["obs_id"], frame["alt_id"])],
        dtype=float,
    )
    assert np.allclose(frame["distance"].to_numpy(dtype=float), expected)


def test_add_interaction_alignment_sampled():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=7,
    )

    idx = pd.MultiIndex.from_product(
        [choosers.index, alternatives.index], names=["obs_id", "alt_id"]
    )
    vals = pd.Series([obs * 10 + alt for obs, alt in idx], index=idx)

    ct2 = ct.add_interaction("distance", vals)
    frame = ct2.to_frame()

    expected = np.asarray(
        [vals.loc[(obs, alt)] for obs, alt in zip(frame["obs_id"], frame["alt_id"])],
        dtype=float,
    )
    assert np.allclose(frame["distance"].to_numpy(dtype=float), expected)


def test_add_interaction_expression_product_broadcasts_sources():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    ct2 = ct.add_interaction_expression(
        "obs_x_alt",
        "obs_feature",
        "alt_feature",
    )
    frame = ct2.to_frame()

    expected = frame["obs_feature"].to_numpy() * frame["alt_feature"].to_numpy()
    assert np.allclose(frame["obs_x_alt"].to_numpy(), expected)


def test_add_interaction_expression_missing_available_rows_raise():
    choosers, alternatives = _make_tables()
    alternatives = alternatives.copy()
    alternatives.loc[11, "alt_feature"] = np.nan
    alternatives["available"] = [1, 1, 1, 1]

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        available="available",
    )

    with pytest.raises(ValueError, match="available alternatives"):
        ct.add_interaction_expression(
            "obs_x_alt",
            "obs_feature",
            "alt_feature",
            missing_policy="allow_unavailable",
        )


def test_add_interaction_expression_allows_missing_unavailable_rows():
    choosers, alternatives = _make_tables()
    alternatives = alternatives.copy()
    alternatives.loc[13, "alt_feature"] = np.nan
    alternatives["available"] = [1, 1, 1, 0]

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        available="available",
    )

    ct2 = ct.add_interaction_expression(
        "obs_x_alt",
        "obs_feature",
        "alt_feature",
        missing_policy="allow_unavailable",
    )

    frame = ct2.to_frame()
    unavailable = frame["available"].to_numpy() == 0
    assert np.isnan(frame.loc[unavailable, "obs_x_alt"]).all()


def test_add_interaction_raises_for_missing_sampled_alt():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=3,
    )

    obs = ct._ds.coords["obs_id"].values[0]
    bad_series = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([(obs, 999)], names=["obs_id", "alt_id"]),
    )

    with pytest.raises(KeyError):
        ct.add_interaction("distance", bad_series)


def test_from_tables_to_frame_and_from_long_round_trip():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    idx = pd.MultiIndex.from_product(
        [choosers.index, alternatives.index], names=["obs_id", "alt_id"]
    )
    vals = pd.Series([obs + alt / 1000.0 for obs, alt in idx], index=idx)

    ct = ct.add_interaction("distance", vals)
    frame = ct.to_frame()

    round_trip = ChoiceTable.from_long(
        frame,
        obs_id_col="obs_id",
        alt_id_col="alt_id",
        choice_col="chosen",
    )

    left = frame.sort_values(["obs_id", "alt_id"]).reset_index(drop=True)
    right = round_trip.to_frame().sort_values(["obs_id", "alt_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_to_arrays_shapes():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=3,
        replace=False,
        seed=10,
    )

    arrays = ct.to_arrays(formula="obs_feature + alt_feature - 1")
    assert arrays.chosen.shape == (ct.n_observations, ct.n_alternatives)


def test_from_tables_interactions_alignment_sampled():
    choosers, alternatives = _make_tables()

    idx = pd.MultiIndex.from_product(
        [choosers.index, alternatives.index], names=["obs_id", "alt_id"]
    )
    distance = pd.Series([obs * 100 + alt for obs, alt in idx], index=idx)

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=21,
        interactions={"distance": distance},
    )

    frame = ct.to_frame()
    expected = np.asarray(
        [distance.loc[(obs, alt)] for obs, alt in zip(frame["obs_id"], frame["alt_id"])],
        dtype=float,
    )
    assert "distance" not in ct._ds.data_vars
    assert np.allclose(frame["distance"].to_numpy(dtype=float), expected)


def test_from_tables_matrix_data_dense_sampled_alignment():
    choosers, alternatives = _make_tables()

    # Dense matrix_data over full alternative universe order [10,11,12,13]
    dense = np.array(
        [
            [100.0, 101.0, 102.0, 103.0],
            [200.0, 201.0, 202.0, 203.0],
            [300.0, 301.0, 302.0, 303.0],
        ]
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=3,
        replace=False,
        seed=33,
        matrix_data={"distance_dense": dense},
    )

    assert "distance_dense" not in ct._ds.data_vars

    frame = ct.to_frame()
    alt_to_pos = {alt_id: i for i, alt_id in enumerate(alternatives.index.to_numpy())}
    expected = np.asarray(
        [
            dense[int(obs) - 1, alt_to_pos[alt]]
            for obs, alt in zip(frame["obs_id"].to_numpy(), frame["alt_id"].to_numpy())
        ],
        dtype=float,
    )
    assert np.allclose(frame["distance_dense"].to_numpy(dtype=float), expected)


def test_from_tables_available_string_column_census():
    choosers, alternatives = _make_tables()
    alternatives = alternatives.copy()
    alternatives["is_open"] = [1, 0, 1, 0]

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        available="is_open",
    )

    frame = ct.to_frame()
    expected = np.asarray([alternatives.loc[alt, "is_open"] for alt in frame["alt_id"]])
    assert ct.available_col == "is_open"
    assert "is_open" in frame.columns
    assert np.array_equal(frame["is_open"].to_numpy(), expected)


def test_from_tables_available_series_1d_alignment_sampled():
    choosers, alternatives = _make_tables()
    avail = pd.Series([1, 0, 1, 1], index=alternatives.index, name="avail_1d")

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=13,
        available=avail,
    )

    frame = ct.to_frame()
    expected = np.asarray([avail.loc[alt] for alt in frame["alt_id"]])
    assert ct.available_col == "avail_1d"
    assert np.array_equal(frame["avail_1d"].to_numpy(), expected)


def test_from_tables_available_series_2d_alignment_sampled():
    choosers, alternatives = _make_tables()

    idx = pd.MultiIndex.from_product(
        [choosers.index, alternatives.index], names=["obs_id", "alt_id"]
    )
    avail = pd.Series(
        [int((obs + alt) % 2 == 0) for obs, alt in idx],
        index=idx,
        name="avail_2d",
    )

    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=3,
        replace=False,
        seed=17,
        available=avail,
    )

    frame = ct.to_frame()
    expected = np.asarray(
        [avail.loc[(obs, alt)] for obs, alt in zip(frame["obs_id"], frame["alt_id"])]
    )
    assert ct.available_col == "avail_2d"
    assert np.array_equal(frame["avail_2d"].to_numpy(), expected)


def test_from_long_non_rectangular_raises():
    df = pd.DataFrame(
        {
            "obs_id": [1, 1, 2],
            "alt_id": [10, 11, 10],
            "x": [1.0, 2.0, 3.0],
            "chosen": [1, 0, 1],
        }
    )

    with pytest.raises(ValueError, match="Non-rectangular long data"):
        ChoiceTable.from_long(df, obs_id_col="obs_id", alt_id_col="alt_id", choice_col="chosen")


def test_from_long_duplicate_pairs_raises():
    df = pd.DataFrame(
        {
            "obs_id": [1, 1],
            "alt_id": [10, 10],
            "x": [1.0, 2.0],
            "chosen": [1, 0],
        }
    )

    with pytest.raises(ValueError, match=r"Duplicate \(obs_id, alt_id\) pairs"):
        ChoiceTable.from_long(df, obs_id_col="obs_id", alt_id_col="alt_id", choice_col="chosen")


def test_to_frame_returns_defensive_copy():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    frame1 = ct.to_frame()
    frame1["_mutated"] = 1

    frame2 = ct.to_frame()
    assert "_mutated" not in frame2.columns


def test_to_dataset_returns_defensive_copy():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    ds = ct.to_dataset()
    ds["_temp"] = (("obs_id", "alt_pos"), np.zeros((ct.n_observations, ct.n_alternatives)))

    assert "_temp" not in ct._ds.data_vars


def test_sampling_design_accessor_sampled():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=9,
    )

    design = ct.sampling_design
    assert design is not None
    assert design["method"] == "srswor"
    assert design["sample_size"] == 2
    assert design["n_alts_full"] == 4
    assert design["replace"] is False
    assert design["seed"] == 9


def test_sampling_design_accessor_census_none():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
    )

    assert ct.sampling_design is None


def test_n_alternatives_full_tracks_pre_sampling_universe():
    choosers, alternatives = _make_tables()
    ct = ChoiceTable.from_tables(
        choosers,
        alternatives,
        chosen_alternatives="chosen_alt",
        sample_size=2,
        replace=False,
        seed=12,
    )

    assert ct.n_alternatives == 2
    assert ct.n_alternatives_full == 4
