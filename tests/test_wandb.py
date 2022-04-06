import json
import os
from unittest.mock import call, patch

import pytest

from wicker.plugins.wandb import (
    _identify_s3_url_for_dataset_version,
    _set_wandb_credentials,
    version_dataset,
)


@pytest.fixture
def temp_config(request, tmpdir):
    # parse the config json into a proper config
    config_json = request.param.get("config_json")
    parsable_config = {
        "wandb_config": config_json.get("wandb_config", {}),
        "aws_s3_config": {
            "s3_datasets_path": config_json.get("aws_s3_config", {}).get("s3_datasets_path", None),
            "region": config_json.get("aws_s3_config", {}).get("region", None),
        },
    }

    # set the config path env and then get the creds from the function
    temp_config = tmpdir / "temp_config.json"
    with open(temp_config, "w") as file_stream:
        json.dump(parsable_config, file_stream)
    os.environ["WICKER_CONFIG_PATH"] = str(temp_config)
    return temp_config, parsable_config


@pytest.mark.parametrize(
    "temp_config, dataset_name, dataset_version, dataset_metadata, entity",
    [
        (
            {
                "config_json": {
                    "aws_s3_config": {
                        "s3_datasets_path": "s3://test_path/to_nowhere/",
                        "region": "us-west-2",
                    },
                    "wandb_config": {
                        "wandb_api_key": "test_key",
                        "wandb_base_url": "test_url",
                    },
                }
            },
            "test_data",
            "0.0.1",
            {"test_key": "test_value"},
            "test_entity",
        )
    ],
    indirect=["temp_config"],
)
def test_version_dataset(temp_config, dataset_name, dataset_version, dataset_metadata, entity):
    """
    GIVEN: A mocked dataset folder to track, a dataset version to track, metadata to track, and a backend to use
    WHEN: This dataset is registered or versioned on W&B
    THEN: The dataset shows up as a run on the portal
    """
    # need to mock out all the wandb calls and test just the inputs to them
    _, config = temp_config
    with patch("wicker.plugins.wandb.wandb.init") as patched_wandb_init:
        with patch("wicker.plugins.wandb.wandb.Artifact") as patched_artifact:
            # version the dataset with the patched functions/classes
            version_dataset(dataset_name, dataset_version, entity, dataset_metadata)

            # establish the expected calls
            expected_artifact_calls = [
                call(f"{dataset_name}_{dataset_version}", type="dataset"),
                call().add_reference(
                    f"{config['aws_s3_config']['s3_datasets_path']}{dataset_name}/{dataset_version}/assets",
                    name="dataset",
                ),
                call().metadata.__setitem__("version", dataset_version),
                call().metadata.__setitem__(
                    "s3_uri", f"{config['aws_s3_config']['s3_datasets_path']}{dataset_name}/{dataset_version}/assets"
                ),
            ]
            for key, value in dataset_metadata.items():
                expected_artifact_calls.append(call().metadata.__setitem__(key, value))

            expected_run_calls = [
                call(project="dataset_curation", name=f"{dataset_name}_{dataset_version}", entity=entity),
                call().log_artifact(patched_artifact()),
            ]

            # assert that these are properly called
            patched_artifact.assert_has_calls(expected_artifact_calls, any_order=True)

            patched_wandb_init.assert_has_calls(expected_run_calls, any_order=True)


@pytest.mark.parametrize(
    "credentials_to_load, temp_config",
    [
        (
            {"WANDB_BASE_URL": "env_base", "WANDB_API_KEY": "env_key", "WANDB_USER_EMAIL": "env_email"},
            {
                "config_json": {
                    "wandb_config": {
                        "wandb_base_url": "config_base",
                        "wandb_api_key": "config_key",
                    }
                }
            },
        )
    ],
    ids=["basic test to override all params"],
    indirect=["temp_config"],
)
def test_set_wandb_credentials(credentials_to_load, temp_config, tmpdir):
    """
    GIVEN: A set of credentials as existing envs and a config json specifying creds
    WHEN: The configs are requested for wandb
    THEN: The proper env variables should be set in the environment based on rules, default to
          wicker config and reject preset env variables
    """
    temp_config_pth, config_json = temp_config

    # load the creds into the env as the base comparison
    for key, value in credentials_to_load.items():
        os.environ[key] = value

    # compare the creds for expected results
    _set_wandb_credentials()
    assert os.environ["WANDB_BASE_URL"] == config_json["wandb_config"]["wandb_base_url"]
    assert os.environ["WANDB_API_KEY"] == config_json["wandb_config"]["wandb_api_key"]


@pytest.mark.parametrize(
    "temp_config, dataset_name, dataset_version, dataset_backend, expected_url",
    [
        (
            {"config_json": {"aws_s3_config": {"s3_datasets_path": "s3://test_path_to_nowhere", "region": "test"}}},
            "test_dataset",
            "0.0.0",
            "s3",
            "s3://test_path_to_nowhere/test_dataset/0.0.0/assets",
        )
    ],
    ids=["Basic test with s3"],
    indirect=["temp_config"],
)
def test_identify_s3_url_for_dataset_version(temp_config, dataset_name, dataset_version, dataset_backend, expected_url):
    """
    GIVEN: A temporary config, a dataset name, version, backend, and expected url
    WHEN: The assets url is pulled from the path factory with these parameters
    THEN: The expected url should match what is returned from the function
    """
    parsed_dataset_url = _identify_s3_url_for_dataset_version(dataset_name, dataset_version, dataset_backend)
    assert parsed_dataset_url == expected_url
