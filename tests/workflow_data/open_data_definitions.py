# flake8: noqa

import os
import sys
import importlib.util
cf_base = os.environ["CF_BASE"]

testdata_path = f"{cf_base}/tests/workflow_data/config_test_files.py"
spec = importlib.util.spec_from_file_location("config_test_files", testdata_path)
foo = importlib.util.module_from_spec(spec)
sys.modules["config_test_files"] = foo
spec.loader.exec_module(foo)


add_config(
    ana,
    foo.campaign_cms_opendata_2015_agc.copy(),
    config_name="test_campaign_limited",
    config_id=2,
    limit_dataset_files=2,
    get_dataset_lfns=foo.get_dataset_lfns,
    get_dataset_lfns_remote_fs=foo.get_dataset_lfns_remote_fs,
    dataset_names=foo.all_datasets,
    process_names=foo.all_processes,
)