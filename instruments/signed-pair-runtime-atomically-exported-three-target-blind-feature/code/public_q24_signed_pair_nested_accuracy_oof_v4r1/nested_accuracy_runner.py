#!/usr/bin/env python3
"""Fail-closed one-shot nested signed-pair accuracy runner V3.

The checked-in authority files are inert scaffolds.  Metadata preflight opens
no manifest, dynamic source, bank, row, weight, or target bytes.  A future live
run requires the acyclic chain

    immutable execution anchor -> one-use capability
    final release              -> anchor + capability

and an independently reviewed lean-V4 post-generation seal for the exact
runtime-V4 three-bank output.  All expensive integrity work occurs only after the unique
claim and same-inode hard-link/unlink capability consumption, and before any
target value is decoded.  The nested folds, J reconstruction, H selection,
metrics, accuracy gates, and target firewall are inherited unchanged.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import sys
import time
from typing import Any


HERE = Path(__file__).absolute().parent
P2 = HERE.parent
V1 = P2 / "public_q24_signed_pair_nested_accuracy_oof_v1"
V2 = P2 / "public_q24_signed_pair_nested_accuracy_oof_v2"
SCIENCE_RUNNER = V1 / "nested_accuracy_runner.py"
V1_HOLD = V1 / "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json"
RUNTIME_V2 = P2 / "public_q24_signed_pair_three_bank_runtime_v2"
RUNTIME_REVIEW_V2 = P2 / "public_q24_signed_pair_three_bank_runtime_review_v2"
RUNTIME_V3 = P2 / "public_q24_signed_pair_three_bank_runtime_v3"
RUNTIME_REVIEW_V3 = P2 / "public_q24_signed_pair_three_bank_runtime_review_v3"
RUNTIME_V4 = P2 / "public_q24_signed_pair_three_bank_runtime_v4"
RUNTIME_REVIEW_V4 = P2 / "public_q24_signed_pair_three_bank_runtime_review_v4"
RUNTIME_V5 = P2 / "public_q24_signed_pair_three_bank_runtime_v5"
RUNTIME_REVIEW_V5 = P2 / "public_q24_signed_pair_three_bank_runtime_review_v5"
V4_INCIDENT_AUDIT = (
    P2
    / "public_q24_signed_pair_lean_v4_flopscope_accumulator_incident_audit_v1"
    / "INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json"
)
V5_AUTHORITY_BUILDER = (
    P2 / "public_q24_signed_pair_lean_v5_prelaunch_authority_builder_v1"
)
V5_AUTHORITY_BUILDER_REVIEW = (
    P2 / "public_q24_signed_pair_lean_v5_prelaunch_authority_builder_review_v1"
)
THREE_BANK_SOURCE_V1 = P2 / "public_q24_signed_pair_three_bank_source_v1"

PREREG = HERE / "PREREGISTRATION.json"
STATIC = HERE / "STATIC_VERIFICATION.json"
ANCHOR_SCAFFOLD = HERE / "EXECUTION_ANCHOR_SCAFFOLD.json"
ANCHOR_REVIEW_SCAFFOLD = HERE / "EXECUTION_ANCHOR_REVIEW_SCAFFOLD.json"
RELEASE_SCAFFOLD = HERE / "EXECUTION_RELEASE_SCAFFOLD.json"
CAPABILITY_SCAFFOLD = HERE / "TARGET_CAPABILITY_SCAFFOLD.json"
GENERATION_AUTHORITY_CONTRACT = HERE / "BANK_SOURCE_AUTHORITY_CONTRACT.json"
POST_GENERATION_SEAL_CONTRACT = HERE / "POST_GENERATION_SEAL_CONTRACT.json"
POST_GENERATION_SEALER = HERE / "post_generation_sealer.py"

ACTION_V2 = P2 / "public_q24_signed_pair_action_state_v2"
NATIVE22 = P2 / "public_q24_signed_pair_qinnovation_distillation_v1"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
QREF_SOURCE = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
RAW_STATS = P2 / "public_full688_dynamic_multistate_residual_cv_v1/cache/raw_sufficient_stats.npz"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
FEATURE_MANIFEST = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/manifest.json"
)

BUNDLE = HERE / "SIGNED_PAIR_NESTED_ACCURACY_OUTPUT_V3"
CLAIM = HERE / "SIGNED_PAIR_NESTED_ACCURACY_RUN_CLAIM_V3.json"
FULL_PREFLIGHT = HERE / "SIGNED_PAIR_NESTED_ACCURACY_FULL_PREFLIGHT_V3.json"
FAILURE = HERE / "SIGNED_PAIR_NESTED_ACCURACY_TERMINAL_FAILURE_V3.json"

V1_REVIEWED_SHA256 = {
    "science_runner": "fe88fac259998251feb014a2607a1a3978eb230e2369061a69778e03edc1dba0",
    "hold_receipt": "2caa33b1f9e9f27fc6ceccacaf068d7ff027b79e74e53448210844318d36e78d",
    "manifest": "820fa24d12299b81256d6c3a51b196aaca83d9e1f4c1d745b117730268ed2f7b",
    "freeze_verification": "5ca646f103f860bf8ae63d00d8c0880a2b26e951f718e66010f441e7a5c017ad",
    "static_verification": "bd6ef6bb230b8f578b4486a0bc15b17cb4d5e6f15ef804c2ddd2b3ac8e3a9b41",
}

V2_REVIEWED_SHA256 = {
    "runner": "6ca6785589053a24ab1ec28502796e52aa888d4b98d21bf9b03c54a5c34da679",
    "manifest": "c393b2379c03538d0510ed05f2b1b19b77358781aaf98cdbc8022aae0bf4289b",
    "independent_review": "527d6f7c0cb1e7795eb5476fd54e135a6a43c757b83de35b6b4a0276decb7ff6",
}

RUNTIME_V2_HASHES = {
    "package_manifest_sha256": "4496721d7c68670db23ba603bf23b9ef6f4d72790b0a9f8beae1c368122880a4",
    "freeze_receipt_sha256": "21db7318d10e948be3b64ebb26106257ce754f7202748033fad2da620e3e38ef",
    "freeze_verification_sha256": "2070f1f009976289ca23638077ca13a9e68d739205d70c20a7f62282af80891c",
    "static_verification_sha256": "f4fb3be6d9b5a15e0bd5f205cb8c49a7543d74768e9c000ce5e03fc7a6849efa",
    "runner_sha256": "cce69e9ed6c4237771f0ebd0023fa4909413988a71efce7d9725f1286f927f83",
    "callable_bundle_manifest_sha256": "190ab5db55f53cdd601768720825a640749c147b40d359f726256f500834b27e",
    "execution_stack_manifest_sha256": "d1251d8f4f2fcf2dddb68df8a0596a7c06fa44525b24420b668cf2193310b009",
    "independent_review_sha256": "7bf6664b2e27d863e5fc4a556373a217d05628ec681141757a4d0beec46c0b03",
}

RUNTIME_V2_PATH_HASHES = {
    RUNTIME_V2 / "PACKAGE_MANIFEST.json": RUNTIME_V2_HASHES[
        "package_manifest_sha256"
    ],
    RUNTIME_V2 / "FREEZE_RECEIPT.json": RUNTIME_V2_HASHES[
        "freeze_receipt_sha256"
    ],
    RUNTIME_V2 / "FREEZE_VERIFICATION.json": RUNTIME_V2_HASHES[
        "freeze_verification_sha256"
    ],
    RUNTIME_V2 / "STATIC_VERIFICATION.json": RUNTIME_V2_HASHES[
        "static_verification_sha256"
    ],
    RUNTIME_V2 / "runtime_runner.py": RUNTIME_V2_HASHES["runner_sha256"],
    RUNTIME_V2 / "CALLABLE_BUNDLE_MANIFEST.json": RUNTIME_V2_HASHES[
        "callable_bundle_manifest_sha256"
    ],
    RUNTIME_V2 / "EXECUTION_STACK_MANIFEST.json": RUNTIME_V2_HASHES[
        "execution_stack_manifest_sha256"
    ],
    RUNTIME_REVIEW_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json": RUNTIME_V2_HASHES[
        "independent_review_sha256"
    ],
}

RUNTIME_V2_COMPONENT_SHA256 = {
    "runtime_token.py": "d75fe8b99a947c37184b8898923f3924a82e171d53ae59e4d9021973e5552333",
    "parent_capture.py": "10ed893bbd50257f21b434fb90e95185026bdf73740b3b6590a001cc1fecc425",
    "signed_pair_literal.py": "20a50cc065c0a9fc3c22fe5b5e178771b2b1fefeacc4a8a68ea48748fb2d2a57",
    "signed_pair_reference.py": "0d9eb525ac5b39d7bdc91ade37b902c82ee819322e44463045419f8e07d0b9a4",
    "capture_adapter.py": "a597cca022323a973a336f2bdda234326bc112fed113fa6870069cadc2ff36e3",
}

RUNTIME_V3_HASHES = {
    "package_manifest_sha256": "a98732cb5f93e7c9b4e5f83e9fa8dfd90dcd2de40e085fcd59388816d6929fe0",
    "freeze_receipt_sha256": "9639f30e3318ce9400e15634981001f04622824d10841ce7e3422501195a604d",
    "freeze_verification_sha256": "c829259337cc14b57764111e7070c3563114e48487ab6c0d269fdbc7596d159a",
    "static_verification_sha256": "d2e802645d1ec7e596608d269ee30a5dddc43bdc8ac7d430574a7b90fd4d9706",
    "runner_sha256": "8c57de66e3758087ca938decdcd9c56f0e8f54d04f8f8a20e940d061c0160cff",
    "callable_bundle_manifest_sha256": "6be8fdc8586416b86e618e77dd707c5d2c828dfd96dbacbcf45629bdcc816f94",
    "execution_stack_manifest_sha256": "e4204ec69ab30a5bb9e441558722edb7bcd80d45bba380ee9cc84217bfa6d433",
    "independent_review_sha256": "5dd705ce746c9808b98b7188210e5ec91e4df300906c4a7b1835a9112c97edf2",
}
RUNTIME_V3_COMPONENT_SHA256 = {
    "runtime_token.py": "d75fe8b99a947c37184b8898923f3924a82e171d53ae59e4d9021973e5552333",
    "parent_capture.py": "10ed893bbd50257f21b434fb90e95185026bdf73740b3b6590a001cc1fecc425",
    "signed_pair_literal.py": "20a50cc065c0a9fc3c22fe5b5e178771b2b1fefeacc4a8a68ea48748fb2d2a57",
    "signed_pair_reference.py": "0d9eb525ac5b39d7bdc91ade37b902c82ee819322e44463045419f8e07d0b9a4",
    "capture_adapter.py": "6d0ed14e7da366cd54aed0fb1a2eb286dc93d0b3009e68bbb644f469a670f362",
}

RUNTIME_V3_PATH_HASHES = {
    RUNTIME_V3 / "PACKAGE_MANIFEST.json": RUNTIME_V3_HASHES[
        "package_manifest_sha256"
    ],
    RUNTIME_V3 / "FREEZE_RECEIPT.json": RUNTIME_V3_HASHES[
        "freeze_receipt_sha256"
    ],
    RUNTIME_V3 / "FREEZE_VERIFICATION.json": RUNTIME_V3_HASHES[
        "freeze_verification_sha256"
    ],
    RUNTIME_V3 / "STATIC_VERIFICATION.json": RUNTIME_V3_HASHES[
        "static_verification_sha256"
    ],
    RUNTIME_V3 / "runtime_runner.py": RUNTIME_V3_HASHES["runner_sha256"],
    RUNTIME_V3 / "CALLABLE_BUNDLE_MANIFEST.json": RUNTIME_V3_HASHES[
        "callable_bundle_manifest_sha256"
    ],
    RUNTIME_V3 / "EXECUTION_STACK_MANIFEST.json": RUNTIME_V3_HASHES[
        "execution_stack_manifest_sha256"
    ],
    RUNTIME_REVIEW_V3 / "INDEPENDENT_STATIC_REVIEW_V3.json": RUNTIME_V3_HASHES[
        "independent_review_sha256"
    ],
}

# V4 is the fresh, narrow manifest-firewall repair after V3 terminated before
# row 0.  Its numerical source law, row/meter schemas, sentinel ranks, and
# seven-file bank bundle are unchanged; the transaction authority is distinct.
RUNTIME_V4_HASHES = {
    "package_manifest_sha256": "2de4795b95f977578be0db33b8cc6a9a2ba634c9a9d7758053d286f899918393",
    "freeze_receipt_sha256": "4e681b90d2d7ba901ed54d5c119cab87c6b7ed500de7e11004ff823ac598f16a",
    "freeze_verification_sha256": "97c5f3d4851e0f1930a8f4a3a697f1c6343d2bc08c5b5022b96189b0e95d0ad7",
    "static_verification_sha256": "00dd27d1ae703095fa78ec6dc67e26568e924aea350d2f5257d58e6ea98ddb7c",
    "runner_sha256": "9cbb4e5057e4ee635372ea0428d94ed563a3a3fbf0f0e6739ae908953b797f66",
    "callable_bundle_manifest_sha256": "5a5c7ea5fd83905ad4484cd74a0101eb076f1c6d2167a423f7000968ef1a2599",
    "execution_stack_manifest_sha256": "e4204ec69ab30a5bb9e441558722edb7bcd80d45bba380ee9cc84217bfa6d433",
    "independent_review_sha256": "b4b717118a0827f9856e7ff675da001eff5c631695cd6c7e343ced24d4d60343",
    "independent_review_request_sha256": "219039c7f53a4660a1ddf7e3a76fd74265f1bde784890dff4d5a1be177a89280",
    "v3_terminal_failure_lineage_sha256": "b0756fd217c6bef5a0df473502109d18bb948fe8b12d0c69c4f15ab02f27ea70",
    "manifest_compatibility_scan_receipt_sha256": "052842efefda1fd73e49d15f17256411436900393cbaf6ae61dac8098310ce0a",
}
RUNTIME_V4_COMPONENT_SHA256 = {
    "runtime_token.py": "d75fe8b99a947c37184b8898923f3924a82e171d53ae59e4d9021973e5552333",
    "parent_capture.py": "10ed893bbd50257f21b434fb90e95185026bdf73740b3b6590a001cc1fecc425",
    "signed_pair_literal.py": "20a50cc065c0a9fc3c22fe5b5e178771b2b1fefeacc4a8a68ea48748fb2d2a57",
    "signed_pair_reference.py": "0d9eb525ac5b39d7bdc91ade37b902c82ee819322e44463045419f8e07d0b9a4",
    "capture_adapter.py": "3888f14a6bc0ad7c3447465265259de0969b59a2a183ec6474310fb12b440d81",
}

RUNTIME_V4_PATH_HASHES = {
    RUNTIME_V4 / "PACKAGE_MANIFEST.json": RUNTIME_V4_HASHES[
        "package_manifest_sha256"
    ],
    RUNTIME_V4 / "FREEZE_RECEIPT.json": RUNTIME_V4_HASHES[
        "freeze_receipt_sha256"
    ],
    RUNTIME_V4 / "FREEZE_VERIFICATION.json": RUNTIME_V4_HASHES[
        "freeze_verification_sha256"
    ],
    RUNTIME_V4 / "STATIC_VERIFICATION.json": RUNTIME_V4_HASHES[
        "static_verification_sha256"
    ],
    RUNTIME_V4 / "runtime_runner.py": RUNTIME_V4_HASHES["runner_sha256"],
    RUNTIME_V4 / "CALLABLE_BUNDLE_MANIFEST.json": RUNTIME_V4_HASHES[
        "callable_bundle_manifest_sha256"
    ],
    RUNTIME_V4 / "EXECUTION_STACK_MANIFEST.json": RUNTIME_V4_HASHES[
        "execution_stack_manifest_sha256"
    ],
    RUNTIME_V4 / "INDEPENDENT_REVIEW_REQUEST.md": RUNTIME_V4_HASHES[
        "independent_review_request_sha256"
    ],
    RUNTIME_V4 / "V3_TERMINAL_FAILURE_LINEAGE_V4.json": RUNTIME_V4_HASHES[
        "v3_terminal_failure_lineage_sha256"
    ],
    RUNTIME_V4 / "MANIFEST_COMPATIBILITY_SCAN_RECEIPT_V4.json": RUNTIME_V4_HASHES[
        "manifest_compatibility_scan_receipt_sha256"
    ],
    RUNTIME_REVIEW_V4 / "INDEPENDENT_STATIC_REVIEW_V4.json": RUNTIME_V4_HASHES[
        "independent_review_sha256"
    ],
}

# Runtime V5 is the fresh successor to terminal/no-retry Runtime V4.  The five
# callable science components, callable bundle, shipping stack, row/bank
# schemas, and generation-receipt schema remain byte/shape identical.  V5 adds
# only the reviewed post-commit FlopScope reset and fresh control-plane lineage.
RUNTIME_V5_HASHES = {
    "package_manifest_sha256": "6307642b0ebe6397e9ffb7c5d163fe29970b5644fb39f76a240fe59afbd4935d",
    "freeze_receipt_sha256": "0e86f6c85e29e9d87da7538694041d65cce63cd7690fd7bba48dd7d8d06ce384",
    "freeze_verification_sha256": "355368e1ed6a4ec0dea54f59bbb9e3273e72e9e674eae825fbfed18f69bceff9",
    "static_verification_sha256": "fb771014f153980f23e4bbe1fe60d1f229bce1b11849dc09464fa747c021f4a7",
    "flopscope_reset_verification_sha256": "9253d354914da94d193dadd43e118ef7b6afaa9a2b11769c8d599cbfd62b26af",
    "runner_sha256": "db30d0743940b654116d94542278b970caa3eece8d9da0d641f46e0291e0c3c4",
    "callable_bundle_manifest_sha256": "5a5c7ea5fd83905ad4484cd74a0101eb076f1c6d2167a423f7000968ef1a2599",
    "execution_stack_manifest_sha256": "e4204ec69ab30a5bb9e441558722edb7bcd80d45bba380ee9cc84217bfa6d433",
    "independent_review_sha256": "26af3b9cf9cb4320b9dfff1917723655ee7ba4046dc99a302fcd2f1608cb2f8e",
    "independent_review_request_sha256": "16f0ea8b369116eaf78f1c33de2d04827ebdcb7712a2e6a1fe30bee04d7fef94",
    "v3_terminal_failure_lineage_sha256": "b0756fd217c6bef5a0df473502109d18bb948fe8b12d0c69c4f15ab02f27ea70",
    "v4_terminal_failure_lineage_sha256": "0dfdfff19826c163958bfc0b6866b31a9f334a7a2f62262a447b5baa3f5d7ded",
    "independent_incident_root_cause_sha256": "060e7bab3094a7929d246f4bdf5fbe6790ccb3aa9c3b05734395a6376b24cd6c",
    "manifest_compatibility_scan_receipt_sha256": "052842efefda1fd73e49d15f17256411436900393cbaf6ae61dac8098310ce0a",
}
RUNTIME_V5_COMPONENT_SHA256 = dict(RUNTIME_V4_COMPONENT_SHA256)
RUNTIME_V5_PATH_HASHES = {
    RUNTIME_V5 / "PACKAGE_MANIFEST.json": RUNTIME_V5_HASHES[
        "package_manifest_sha256"
    ],
    RUNTIME_V5 / "FREEZE_RECEIPT.json": RUNTIME_V5_HASHES[
        "freeze_receipt_sha256"
    ],
    RUNTIME_V5 / "FREEZE_VERIFICATION.json": RUNTIME_V5_HASHES[
        "freeze_verification_sha256"
    ],
    RUNTIME_V5 / "STATIC_VERIFICATION.json": RUNTIME_V5_HASHES[
        "static_verification_sha256"
    ],
    RUNTIME_V5 / "FLOPSCOPE_RESET_VERIFICATION_V5.json": RUNTIME_V5_HASHES[
        "flopscope_reset_verification_sha256"
    ],
    RUNTIME_V5 / "runtime_runner.py": RUNTIME_V5_HASHES["runner_sha256"],
    RUNTIME_V5 / "CALLABLE_BUNDLE_MANIFEST.json": RUNTIME_V5_HASHES[
        "callable_bundle_manifest_sha256"
    ],
    RUNTIME_V5 / "EXECUTION_STACK_MANIFEST.json": RUNTIME_V5_HASHES[
        "execution_stack_manifest_sha256"
    ],
    RUNTIME_V5 / "INDEPENDENT_REVIEW_REQUEST.md": RUNTIME_V5_HASHES[
        "independent_review_request_sha256"
    ],
    RUNTIME_V5 / "V3_TERMINAL_FAILURE_LINEAGE_V4.json": RUNTIME_V5_HASHES[
        "v3_terminal_failure_lineage_sha256"
    ],
    RUNTIME_V5 / "V4_TERMINAL_FAILURE_LINEAGE_V5.json": RUNTIME_V5_HASHES[
        "v4_terminal_failure_lineage_sha256"
    ],
    RUNTIME_V5 / "MANIFEST_COMPATIBILITY_SCAN_RECEIPT_V4.json": RUNTIME_V5_HASHES[
        "manifest_compatibility_scan_receipt_sha256"
    ],
    RUNTIME_REVIEW_V5 / "INDEPENDENT_STATIC_REVIEW_V5.json": RUNTIME_V5_HASHES[
        "independent_review_sha256"
    ],
    V4_INCIDENT_AUDIT: RUNTIME_V5_HASHES[
        "independent_incident_root_cause_sha256"
    ],
}
V5_AUTHORITY_BUILDER_HASHES = {
    "authority_builder_bindings_sha256": (
        "697dc4bc7c5afa22c57cec2eb7e9ca48750a1d922751e65f1cc0ac955c32d09a"
    ),
    "authority_builder_freeze_receipt_sha256": (
        "5454acd83ba31b679992d4072620d5889fe2b7f0b7a66f45fb6ae8681cb4859a"
    ),
    "authority_builder_freeze_verification_sha256": (
        "3edad8c6555736c95df5e2eecbc4908fb1e38096eb1cb109f5cb4ffdc0a2827b"
    ),
    "authority_builder_helper_sha256": (
        "bb28d135da80b7ddc3472b16e3a2097da36747b6dec52e0e9cb194c6fa8e509c"
    ),
    "authority_builder_independent_review_sha256": (
        "89b29a7838903d88957f0eb2397f0ef7c88821b50f3c5995cdbccfe57ef79565"
    ),
    "authority_builder_package_manifest_sha256": (
        "3cf0055ac05ea86bf380ecc5c5100d0665308ae222ce15824ffb10d963a2b9d8"
    ),
    "authority_builder_static_verification_sha256": (
        "b8be3e8f34f90126b530381ec55bc286c35b9aca4b7edb07baddcd1f7f59f366"
    ),
}
V5_AUTHORITY_BUILDER_PATH_HASHES = {
    V5_AUTHORITY_BUILDER / "BINDINGS_HOLD.json": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_bindings_sha256"
    ],
    V5_AUTHORITY_BUILDER / "FREEZE_RECEIPT.json": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_freeze_receipt_sha256"
    ],
    V5_AUTHORITY_BUILDER / "FREEZE_VERIFICATION.json": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_freeze_verification_sha256"
    ],
    V5_AUTHORITY_BUILDER / "prelaunch_authority_v5.py": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_helper_sha256"
    ],
    V5_AUTHORITY_BUILDER_REVIEW
    / "INDEPENDENT_STATIC_REVIEW_V1.json": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_independent_review_sha256"
    ],
    V5_AUTHORITY_BUILDER / "PACKAGE_MANIFEST.json": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_package_manifest_sha256"
    ],
    V5_AUTHORITY_BUILDER / "STATIC_VERIFICATION.json": V5_AUTHORITY_BUILDER_HASHES[
        "authority_builder_static_verification_sha256"
    ],
}

# The receipt profile is deliberately isolated from the frozen nested-OOF
# science body.  A reviewed successor runtime can add one exact profile
# validator and change this single frozen selector without altering folds,
# H-selection, J reconstruction, metrics, or the target firewall.
GENERATION_RECEIPT_PROFILE = "EXACT_LEAN_RUNTIME_V5"

# Exact, separately issued pre-generation authority chain for the one lean V4
# bank transaction.  These are distinct from the runtime package review and
# from the later post-generation seal/review.
LEAN_SOURCE_AUTHORITY_SHA256 = (
    "7c9979864fdeefd2ef7569445c817498d14c8818fa9c7c575d016c4373dd39b5"
)
LEAN_SOURCE_AUTHORITY_REVIEW_SHA256 = (
    "39a697f28096a9dfa9fad6d437bd58ff96a50d64ca7c65c8513c1c7fd2f02922"
)
LEAN_SUPERSESSION_RECORD_SHA256 = (
    "45e00742ad736c0406aa40307991dc23313f7633fc218981621896936c15a225"
)
# These three V4 authority roots were injected only after an independent live
# review.  The predecessor package used non-SHA sentinels so every protected-path
# validation failed closed while that review was pending.
V4_SOURCE_AUTHORITY_SHA256 = (
    "7e6ed0cb40e10497ee0d2ed93f0c0e1430ff056e7c481daef071dfb961493dee"
)
V4_SOURCE_AUTHORITY_REVIEW_SHA256 = (
    "68dde823b5c333bcc4c5226d55e01e207889efe1fb82b7707b8c94ba55acfa00"
)
V4_SUCCESSOR_RECORD_SHA256 = (
    "836f6d5c71cfbf7ad488c086bf384eb3282561334783ffdf05e38a0a85b90978"
)
V3_TERMINAL_FAILURE_BINDINGS = {
    "exclusive_claim_sha256": (
        "bfa9e8d800a5b1efc0ec4e47a62b53d70176e0bf8b9ec73f9981056974abd4e9"
    ),
    "terminal_failure_sha256": (
        "e64f998fc1513f906e90d2869340e44ce2ca548b91e31667aade69da9af1d50d"
    ),
    "consumed_capability_sha256": (
        "f7cd31ab124c7914d1bd2b413289fb84ced4a38d2e93e3f3d3d9a9e711d38a39"
    ),
    "no_retry": True,
    "published_bundle": False,
}
V4_TERMINAL_FAILURE_BINDINGS = {
    "exclusive_claim_sha256": (
        "d690aa5b42dd0bca12705795004640413368a4384bce467bad2a16c6560fd2b7"
    ),
    "terminal_failure_sha256": (
        "fbac37a97c64983fbc56627b67fdebb59dca07d0f43f3762798495022ed74dae"
    ),
    "consumed_capability_sha256": (
        "24db92e64b4129d3a8c002620017aa516e2287cb9183c5edf9c99ec35c6c4518"
    ),
    "independent_incident_root_cause_sha256": RUNTIME_V5_HASHES[
        "independent_incident_root_cause_sha256"
    ],
    "no_retry": True,
    "published_bundle": False,
}

# Fresh pre-generation V5 roots, injected only after the separate builder GO
# and live source-authority review.  The generation receipt/banks/seal remain
# future execution-anchor inputs and are not package-freeze dependencies.
V5_SOURCE_AUTHORITY_SHA256 = (
    "fe74c7264767fda21c566e2cd993e96c5a61b2e5c5332734a8dbfe884b06d785"
)
V5_SOURCE_AUTHORITY_REVIEW_SHA256 = (
    "cff0066295f7b432868719e9ad274ef4a47dc0356889376df461ce3993551fef"
)
V5_SUCCESSOR_RECORD_SHA256 = (
    "4dcaa9865195caab75d8c6217ea51b94a27cc3c008c65212bed47d3b4293f42a"
)
PRIOR_V2_NONREUSE_BINDINGS = {
    "prior_v2_authority_anchor_sha256": (
        "d763881b400dd3905e774c4f26674c823e8fab5861a8e47e60ee16c4ebbb2ec8"
    ),
    "prior_v2_single_use_capability_sha256": (
        "3c14ff5897f5e6f9725d0378fecd073c813fdc5533bffc50e58a68340be5fb61"
    ),
    "prior_v2_final_release_sha256": (
        "9732f5358571bb49dc65026f79b01bc1405ce864630348df429ff7b180c1189d"
    ),
    "prior_v2_source_authority_sha256": (
        "89004154b98574c2a583e8c0ab1fb67717933c25853c59cbaef269920b1bf79c"
    ),
    "prior_v2_live_source_review_sha256": (
        "0751daef0965ed12898483e7eb296aff4ff794d22b666a6d4c132e2f89c1c3dd"
    ),
}

SOURCE_V1_HASHES = {
    "three_bank_source_manifest_sha256": "ab3bc3b700f29dedb27c073c9cf8db995f1f0f275c31cc23c162810b2fbd3e90",
    "three_bank_source_freeze_verification_sha256": "708889d5db98ba27c642f15488bfd2551765dfc266472f9bd22ce4d7415fa6ea",
    "three_bank_source_independent_review_sha256": "e5bbea1e38a9c299ba7f708d7a2160c638473c86a3059970f516b535d26f1b4d",
}

SOURCE_V1_PATH_HASHES = {
    THREE_BANK_SOURCE_V1 / "PACKAGE_MANIFEST.json": SOURCE_V1_HASHES[
        "three_bank_source_manifest_sha256"
    ],
    THREE_BANK_SOURCE_V1 / "FREEZE_VERIFICATION.json": SOURCE_V1_HASHES[
        "three_bank_source_freeze_verification_sha256"
    ],
    THREE_BANK_SOURCE_V1 / "INDEPENDENT_STATIC_REVIEW_V1.json": SOURCE_V1_HASHES[
        "three_bank_source_independent_review_sha256"
    ],
}

ORDERED_SCOPE_BINDINGS = {
    "ordered_688_row_scope_sha256": "2756e969a52dea280fdac2c47fe44713c0706896d5b4dc14d834e3c3c7d423f0",
    "ordered_688_record_sha256": "902466f86013e2bf7e597518d124e4d98a5654fa63949ca8068522383d95bab3",
    "ordered_ranks_int32_sha256": "a4f610ffb6ff6479212f61f4a8d28c0cb308712aff840a8fc42dec6cc98914a4",
    "row_count": 688,
}

PREDECESSOR_HASHES = {
    ACTION_V2 / "PACKAGE_MANIFEST.json": "85c1f3073ef54fa822748c8c579bd1d68d0823bb1169014f92d6e05152c5f895",
    ACTION_V2 / "FREEZE_VERIFICATION.json": "e0846f84c452a1d99e3e296dc4755a28f2b985e18b7e5849414a4d64d4c47490",
    ACTION_V2 / "STATIC_VERIFICATION.json": "c726d844fab90ffca968e53ded72dd19d30a73c169ca51fb2015eb64dd1a3741",
    ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json": "2a9270595bb33c0fd0f17a347131f02358a2aee00f0f38c6ee9e8484b0c45a0b",
    NATIVE22 / "NATIVE22_CAPACITY_OUTPUT_V1/RESULT.json": "5c940c501c383fe65e9e4b5dda103ba287797d0cb18ed22bc6314588f37245fd",
    NATIVE22 / "NATIVE22_TERMINAL_AUDIT_V1.json": "5f3bcd094257a644c2a9ecb243d6bc357634b711be8d40ee3089a1d75cd8a62f",
}

DYNAMIC_HASHES = {
    "J_source_sha256": "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810",
    "QREF_source_sha256": "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2",
    "raw_sufficient_stats_sha256": "1648a140768cdc254a7bb6957dcb8e82f4271118bc64b552bb08de294ec62208",
    "feature_manifest_sha256": "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50",
    "target_bundle_sha256": "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7",
    "native_row_aggregate_sha256": "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3",
    "outer_fold_assignment_uint8_sha256": "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c",
}

ROWS = 688
WIDTH = 1024
DEPTH = 16
OUTER_FOLDS = 4
OUTER_FIT = 516
OUTER_HELD = 172
HS = (16, 64, 256)
PAIR_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
ACTION_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
RAW_HURDLE = 1.2425668176231516e-8
TIE_TOLERANCE = 0.0025
EXACT_BANK_BYTES = {16: 586_153_984, 64: 2_344_615_936, 256: 9_378_463_744}
EXACT_ZERO_BYTES = {16: 65_536, 64: 262_144, 256: 1_048_576}

ACTIVE_CLAIM_NONCE: str | None = None
ACTIVE_TEMP_BUNDLE: Path | None = None


class AuthorityError(RuntimeError):
    """A fail-closed authority, provenance, schema, or integrity rejection."""


@dataclass(frozen=True)
class PathIdentity:
    unresolved: Path
    resolved: Path
    device: int
    inode: int
    size: int
    links: int
    mode: int

    def claim_record(self) -> dict[str, Any]:
        return {
            "unresolved_path": str(self.unresolved),
            "resolved_path": str(self.resolved),
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "links": self.links,
        }


@dataclass(frozen=True)
class ExecutionAuthority:
    release_identity: PathIdentity
    release_raw: bytes
    release: dict[str, Any]
    anchor_identity: PathIdentity
    anchor_raw: bytes
    anchor: dict[str, Any]
    anchor_review_identity: PathIdentity
    capability_identity: PathIdentity
    capability_raw: bytes
    capability: dict[str, Any]
    external_files: dict[str, PathIdentity]
    bank_root: PathIdentity
    preclaim_access: dict[str, Any]


@dataclass(frozen=True)
class BankAuthority:
    receipt_path: Path
    receipt_sha256: str
    ordered_record_sha256: str
    mlp_ids: Any
    bank_paths: dict[int, Path]
    zero_paths: dict[int, Path]
    resource_observations: Any
    novelty_diagnostics: Any
    integrity_receipt: dict[str, Any]


ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def is_sha(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_int(value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise AuthorityError("expected exact non-bool JSON integer")
    if minimum is not None and value < minimum:
        raise AuthorityError("integer below bound")
    if maximum is not None and value > maximum:
        raise AuthorityError("integer above bound")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise AuthorityError(f"{label} fields are not exact")
    return value


def canonical_id(value: Any, label: str, *, exact: str | None = None) -> str:
    if type(value) is not str or ID_RE.fullmatch(value) is None:
        raise AuthorityError(f"{label} is not canonical printable ASCII")
    if exact is not None and value != exact:
        raise AuthorityError(f"{label} issuer mismatch")
    return value


def canonical_utc(value: Any, label: str = "issued_utc") -> str:
    if type(value) is not str:
        raise AuthorityError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise AuthorityError(f"{label} must be canonical UTC") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise AuthorityError(f"{label} is not canonical UTC")
    return value


def validate_ordered_ranks(value: Any) -> None:
    if type(value) is not list or len(value) != ROWS:
        raise AuthorityError("ordered_ranks must contain exactly 688 items")
    for expected, observed in enumerate(value):
        if type(observed) is not int or observed != expected:
            raise AuthorityError("ordered_ranks must be exact non-bool integers 0..687")


def validate_sha_map(value: Any, label: str) -> dict[str, str]:
    mapping = exact_keys(value, {"16", "64", "256"}, label)
    if any(not is_sha(mapping[key]) for key in ("16", "64", "256")):
        raise AuthorityError(f"{label} contains a non-SHA256 value")
    return mapping


def canonical_relative_path(value: Any, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise AuthorityError(f"{label} must be a nonempty canonical relative path")
    if (
        any(ord(character) < 32 or ord(character) > 126 for character in value)
        or "\\" in value
    ):
        raise AuthorityError(f"{label} contains non-printable characters")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or any(part in ("", ".", "..") for part in path.parts):
        raise AuthorityError(f"{label} is not a canonical contained relative path")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat_without_symlink_components(path: Path) -> os.stat_result:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    result: os.stat_result | None = None
    for part in absolute.parts[1:]:
        current = current / part
        try:
            result = os.lstat(current)
        except FileNotFoundError as error:
            raise AuthorityError(f"required path absent: {current}") from error
        if stat.S_ISLNK(result.st_mode):
            raise AuthorityError(f"symlink component rejected before resolve: {current}")
    if result is None:
        raise AuthorityError("filesystem root is not an admissible artifact")
    return result


def secure_file(path: Path, *, expected_size: int | None = None) -> PathIdentity:
    unresolved = _lexical_absolute(path)
    metadata = _lstat_without_symlink_components(unresolved)
    if not stat.S_ISREG(metadata.st_mode):
        raise AuthorityError(f"regular file required: {unresolved}")
    if metadata.st_nlink != 1:
        raise AuthorityError(f"hard-linked file rejected: {unresolved}")
    if expected_size is not None and metadata.st_size != expected_size:
        raise AuthorityError(f"exact byte count mismatch: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    return PathIdentity(
        unresolved=unresolved,
        resolved=resolved,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        links=metadata.st_nlink,
        mode=metadata.st_mode,
    )


def secure_directory(path: Path) -> PathIdentity:
    unresolved = _lexical_absolute(path)
    metadata = _lstat_without_symlink_components(unresolved)
    if not stat.S_ISDIR(metadata.st_mode):
        raise AuthorityError(f"directory required: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    return PathIdentity(
        unresolved=unresolved,
        resolved=resolved,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        links=metadata.st_nlink,
        mode=metadata.st_mode,
    )


def secure_relative_file(
    root: PathIdentity, relative: Any, *, expected_size: int | None = None
) -> PathIdentity:
    text = canonical_relative_path(relative, "relative file path")
    identity = secure_file(root.unresolved / text, expected_size=expected_size)
    if identity.resolved == root.resolved or root.resolved not in identity.resolved.parents:
        raise AuthorityError("relative file escaped its authenticated root")
    return identity


def secure_relative_directory(root: PathIdentity, relative: Any) -> PathIdentity:
    text = canonical_relative_path(relative, "relative directory path")
    identity = secure_directory(root.unresolved / text)
    if identity.resolved == root.resolved or root.resolved not in identity.resolved.parents:
        raise AuthorityError("relative directory escaped its authenticated root")
    return identity


def _open_identity(identity: PathIdentity) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(identity.unresolved, flags)
    observed = os.fstat(descriptor)
    if (
        observed.st_dev != identity.device
        or observed.st_ino != identity.inode
        or observed.st_size != identity.size
        or observed.st_nlink != 1
        or not stat.S_ISREG(observed.st_mode)
    ):
        os.close(descriptor)
        raise AuthorityError(f"artifact identity changed: {identity.unresolved}")
    return descriptor


def read_identity_bytes(identity: PathIdentity) -> bytes:
    descriptor = _open_identity(identity)
    try:
        chunks: list[bytes] = []
        remaining = identity.size
        while remaining:
            block = os.read(descriptor, min(8 << 20, remaining))
            if not block:
                raise AuthorityError(f"short read: {identity.unresolved}")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise AuthorityError(f"artifact grew while read: {identity.unresolved}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def hash_identity(identity: PathIdentity) -> str:
    descriptor = _open_identity(identity)
    digest = hashlib.sha256()
    try:
        remaining = identity.size
        while remaining:
            block = os.read(descriptor, min(8 << 20, remaining))
            if not block:
                raise AuthorityError(f"short hash read: {identity.unresolved}")
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise AuthorityError(f"artifact grew while hashed: {identity.unresolved}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_canonical_json(identity: PathIdentity, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = read_identity_bytes(identity)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuthorityError(f"nonfinite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError(f"invalid canonical JSON: {label}") from error
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise AuthorityError(f"{label} is not exact canonical JSON")
    return raw, value


def write_exclusive_durable(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte exclusive write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def observation(label: str, started: float | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": label,
        "maximum_process_rss_bytes_observational_only": resident_bytes(),
    }
    if started is not None:
        record["elapsed_seconds_observational_only"] = time.perf_counter() - started
    return record


ANCHOR_KEYS = {
    "schema",
    "status",
    "execution_authorized",
    "authority_id",
    "issued_by",
    "issued_utc",
    "package_bindings",
    "predecessor_bindings",
    "dynamic_input_bindings",
    "bank_bindings",
    "role",
    "target_columns",
    "eligible_H",
    "one_shot",
    "retry",
    "observer_selection_targets",
    "names",
    "scores",
    "board",
    "network",
    "package",
    "submission",
}
PACKAGE_BINDING_KEYS = {
    "preregistration_sha256",
    "runner_sha256",
    "static_verification_sha256",
    "post_generation_sealer_sha256",
    "v1_science_runner_sha256",
    "v1_hold_receipt_sha256",
    "v2_runner_sha256",
    "v2_manifest_sha256",
    "v2_independent_review_sha256",
    "bank_source_authority_contract_sha256",
    "post_generation_seal_contract_sha256",
}
PREDECESSOR_BINDING_KEYS = {
    "signed_pair_v2_manifest_sha256",
    "signed_pair_v2_freeze_verification_sha256",
    "signed_pair_v2_static_verification_sha256",
    "signed_pair_v2_independent_review_sha256",
    "native22_result_sha256",
    "native22_terminal_audit_sha256",
    "v1_manifest_sha256",
    "v1_freeze_verification_sha256",
    "v1_static_verification_sha256",
    "v1_hold_receipt_sha256",
}
DYNAMIC_BINDING_KEYS = set(DYNAMIC_HASHES)
BANK_BINDING_KEYS = {
    "source_authority_sha256",
    "source_authority_review_sha256",
    "pre_generation_v4_terminal_successor_record_sha256",
    "v4_terminal_failure_lineage_sha256",
    "independent_incident_root_cause_sha256",
    "flopscope_reset_verification_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "v4_exclusive_claim_sha256",
    "v4_terminal_failure_sha256",
    "v4_consumed_capability_sha256",
    "lean_runtime_v5_package_manifest_sha256",
    "lean_runtime_v5_independent_review_sha256",
    "generation_execution_anchor_sha256",
    "generation_final_release_sha256",
    "generation_single_use_capability_sha256",
    "generation_claim_sha256",
    "post_generation_seal_sha256",
    "post_generation_seal_review_sha256",
    "generation_receipt_sha256",
    "ordered_688_record_sha256",
    "complete_bank_sha256",
    "A15_zero_sha256",
}
ANCHOR_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "execution_anchor_sha256",
    "runner_sha256",
    "preregistration_sha256",
    "static_verification_sha256",
    "reviewed_before_capability_and_release",
    "acyclic_authority_chain_verified",
    "post_generation_independent_seal_required",
    "execution_authorized",
}
CAPABILITY_KEYS = {
    "schema",
    "status",
    "execution_authorized",
    "capability_id",
    "issued_by",
    "issued_utc",
    "execution_anchor_sha256",
    "target_bundle_sha256",
    "ordered_688_record_sha256",
    "outer_fold_assignment_uint8_sha256",
    "role",
    "target_columns",
    "one_use",
    "retry",
    "observer_selection_targets",
    "names",
    "scores",
    "board",
    "network",
    "package",
    "submission",
}
RELEASE_KEYS = {
    "schema",
    "status",
    "execution_authorized",
    "release_id",
    "issued_by",
    "issued_utc",
    "execution_anchor_path",
    "execution_anchor_sha256",
    "execution_anchor_review_path",
    "execution_anchor_review_sha256",
    "single_use_target_capability_path",
    "single_use_target_capability_sha256",
    "bank_source_authority_path",
    "bank_source_authority_review_path",
    "bank_post_generation_seal_path",
    "bank_post_generation_seal_review_path",
    "bank_generation_receipt_path",
    "bank_root",
    "role",
    "target_columns",
    "one_shot",
    "retry",
    "observer_selection_targets",
    "names",
    "scores",
    "board",
    "network",
    "package",
    "submission",
}


def _require_firewall(document: dict[str, Any]) -> None:
    for key in (
        "observer_selection_targets",
        "names",
        "scores",
        "board",
        "network",
        "package",
        "submission",
    ):
        if document.get(key) is not False:
            raise AuthorityError(f"authority firewall changed: {key}")


def validate_anchor_object(
    anchor: Any, *, expected_package: dict[str, str] | None = None
) -> dict[str, Any]:
    document = exact_keys(anchor, ANCHOR_KEYS, "execution anchor")
    if (
        document["schema"] != "public-q24-signed-pair-nested-accuracy-execution-anchor-v3"
        or document["status"] != "IMMUTABLE_ONE_SHOT_ACCURACY_AUTHORITY_ANCHOR"
        or document["execution_authorized"] is not True
    ):
        raise AuthorityError("execution anchor schema/status/authorization mismatch")
    canonical_id(document["authority_id"], "authority_id")
    canonical_id(document["issued_by"], "issued_by", exact="root")
    canonical_utc(document["issued_utc"])
    package_bindings = exact_keys(
        document["package_bindings"], PACKAGE_BINDING_KEYS, "anchor package bindings"
    )
    predecessor_bindings = exact_keys(
        document["predecessor_bindings"],
        PREDECESSOR_BINDING_KEYS,
        "anchor predecessor bindings",
    )
    dynamic_bindings = exact_keys(
        document["dynamic_input_bindings"],
        DYNAMIC_BINDING_KEYS,
        "anchor dynamic bindings",
    )
    bank_bindings = exact_keys(
        document["bank_bindings"], BANK_BINDING_KEYS, "anchor bank bindings"
    )
    for mapping, label in (
        (package_bindings, "package"),
        (predecessor_bindings, "predecessor"),
        (dynamic_bindings, "dynamic"),
    ):
        if any(not is_sha(value) for value in mapping.values()):
            raise AuthorityError(f"anchor {label} binding is not SHA256")
    for key in BANK_BINDING_KEYS - {"complete_bank_sha256", "A15_zero_sha256"}:
        if not is_sha(bank_bindings[key]):
            raise AuthorityError(f"anchor bank binding invalid: {key}")
    validate_sha_map(bank_bindings["complete_bank_sha256"], "complete bank hashes")
    validate_sha_map(bank_bindings["A15_zero_sha256"], "A15 hashes")
    if expected_package is not None and package_bindings != expected_package:
        raise AuthorityError("anchor package bytes differ from frozen repair")
    expected_predecessors = {
        "signed_pair_v2_manifest_sha256": PREDECESSOR_HASHES[ACTION_V2 / "PACKAGE_MANIFEST.json"],
        "signed_pair_v2_freeze_verification_sha256": PREDECESSOR_HASHES[ACTION_V2 / "FREEZE_VERIFICATION.json"],
        "signed_pair_v2_static_verification_sha256": PREDECESSOR_HASHES[ACTION_V2 / "STATIC_VERIFICATION.json"],
        "signed_pair_v2_independent_review_sha256": PREDECESSOR_HASHES[ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"],
        "native22_result_sha256": PREDECESSOR_HASHES[NATIVE22 / "NATIVE22_CAPACITY_OUTPUT_V1/RESULT.json"],
        "native22_terminal_audit_sha256": PREDECESSOR_HASHES[NATIVE22 / "NATIVE22_TERMINAL_AUDIT_V1.json"],
        "v1_manifest_sha256": V1_REVIEWED_SHA256["manifest"],
        "v1_freeze_verification_sha256": V1_REVIEWED_SHA256["freeze_verification"],
        "v1_static_verification_sha256": V1_REVIEWED_SHA256["static_verification"],
        "v1_hold_receipt_sha256": V1_REVIEWED_SHA256["hold_receipt"],
    }
    if predecessor_bindings != expected_predecessors:
        raise AuthorityError("anchor predecessor bindings changed")
    if dynamic_bindings != DYNAMIC_HASHES:
        raise AuthorityError("anchor dynamic input bindings changed")
    if (
        bank_bindings["source_authority_sha256"]
        != V5_SOURCE_AUTHORITY_SHA256
        or bank_bindings["source_authority_review_sha256"]
        != V5_SOURCE_AUTHORITY_REVIEW_SHA256
        or bank_bindings["pre_generation_v4_terminal_successor_record_sha256"]
        != V5_SUCCESSOR_RECORD_SHA256
        or bank_bindings["v4_terminal_failure_lineage_sha256"]
        != RUNTIME_V5_HASHES["v4_terminal_failure_lineage_sha256"]
        or bank_bindings["independent_incident_root_cause_sha256"]
        != RUNTIME_V5_HASHES["independent_incident_root_cause_sha256"]
        or bank_bindings["flopscope_reset_verification_sha256"]
        != RUNTIME_V5_HASHES["flopscope_reset_verification_sha256"]
        or bank_bindings["manifest_compatibility_scan_receipt_sha256"]
        != RUNTIME_V5_HASHES["manifest_compatibility_scan_receipt_sha256"]
        or bank_bindings["v4_exclusive_claim_sha256"]
        != V4_TERMINAL_FAILURE_BINDINGS["exclusive_claim_sha256"]
        or bank_bindings["v4_terminal_failure_sha256"]
        != V4_TERMINAL_FAILURE_BINDINGS["terminal_failure_sha256"]
        or bank_bindings["v4_consumed_capability_sha256"]
        != V4_TERMINAL_FAILURE_BINDINGS["consumed_capability_sha256"]
        or bank_bindings["lean_runtime_v5_package_manifest_sha256"]
        != RUNTIME_V5_HASHES["package_manifest_sha256"]
        or bank_bindings["lean_runtime_v5_independent_review_sha256"]
        != RUNTIME_V5_HASHES["independent_review_sha256"]
    ):
        raise AuthorityError("anchor lean source/runtime-V5/V4-terminal binding changed")
    if (
        document["role"] != "observer_fit"
        or document["target_columns"] != ["mlp_id", "all_layer_means"]
        or document["eligible_H"] != [16, 64, 256]
        or document["one_shot"] is not True
        or document["retry"] is not False
    ):
        raise AuthorityError("anchor role, H set, or transaction law changed")
    _require_firewall(document)
    return document


def validate_anchor_review_object(
    review: Any, *, anchor_sha256: str, package_bindings: dict[str, str]
) -> dict[str, Any]:
    document = exact_keys(review, ANCHOR_REVIEW_KEYS, "execution anchor review")
    if (
        document["schema"]
        != "public-q24-signed-pair-nested-accuracy-execution-anchor-independent-review-v3"
        or document["status"] != "GO_EXACT_ONE_SHOT_ACCURACY_ANCHOR"
        or document["execution_anchor_sha256"] != anchor_sha256
        or document["runner_sha256"] != package_bindings["runner_sha256"]
        or document["preregistration_sha256"]
        != package_bindings["preregistration_sha256"]
        or document["static_verification_sha256"]
        != package_bindings["static_verification_sha256"]
    ):
        raise AuthorityError("execution anchor independent review mismatch")
    canonical_id(document["review_id"], "review_id")
    canonical_id(document["issued_by"], "issued_by", exact="independent_reviewer")
    canonical_utc(document["issued_utc"])
    for key in (
        "reviewed_before_capability_and_release",
        "acyclic_authority_chain_verified",
        "post_generation_independent_seal_required",
        "execution_authorized",
    ):
        if document[key] is not True:
            raise AuthorityError(f"anchor review boolean mismatch: {key}")
    return document


def validate_capability_object(
    capability: Any, *, anchor_sha256: str, bank_bindings: dict[str, Any]
) -> dict[str, Any]:
    document = exact_keys(capability, CAPABILITY_KEYS, "target capability")
    if (
        document["schema"]
        != "public-q24-signed-pair-nested-accuracy-target-capability-v3"
        or document["status"]
        != "AUTHORIZE_EXACTLY_ONE_OBSERVER_FIT_NESTED_ACCURACY_RUN"
        or document["execution_authorized"] is not True
        or document["execution_anchor_sha256"] != anchor_sha256
        or document["target_bundle_sha256"] != DYNAMIC_HASHES["target_bundle_sha256"]
        or document["ordered_688_record_sha256"]
        != bank_bindings["ordered_688_record_sha256"]
        or document["outer_fold_assignment_uint8_sha256"]
        != DYNAMIC_HASHES["outer_fold_assignment_uint8_sha256"]
        or document["role"] != "observer_fit"
        or document["target_columns"] != ["mlp_id", "all_layer_means"]
        or document["one_use"] is not True
        or document["retry"] is not False
    ):
        raise AuthorityError("target capability contract mismatch")
    canonical_id(document["capability_id"], "capability_id")
    canonical_id(document["issued_by"], "issued_by", exact="root")
    canonical_utc(document["issued_utc"])
    _require_firewall(document)
    return document


def validate_release_object(
    release: Any,
    *,
    anchor_sha256: str,
    anchor_review_sha256: str,
    capability_sha256: str,
) -> dict[str, Any]:
    document = exact_keys(release, RELEASE_KEYS, "final execution release")
    if (
        document["schema"]
        != "public-q24-signed-pair-nested-accuracy-execution-release-v3"
        or document["status"] != "RELEASE_EXACTLY_ONE_NESTED_ACCURACY_RUN"
        or document["execution_authorized"] is not True
        or document["execution_anchor_sha256"] != anchor_sha256
        or document["execution_anchor_review_sha256"] != anchor_review_sha256
        or document["single_use_target_capability_sha256"] != capability_sha256
        or document["role"] != "observer_fit"
        or document["target_columns"] != ["mlp_id", "all_layer_means"]
        or document["one_shot"] is not True
        or document["retry"] is not False
    ):
        raise AuthorityError("final execution release contract mismatch")
    canonical_id(document["release_id"], "release_id")
    canonical_id(document["issued_by"], "issued_by", exact="root")
    canonical_utc(document["issued_utc"])
    for key in (
        "execution_anchor_path",
        "execution_anchor_review_path",
        "single_use_target_capability_path",
        "bank_source_authority_path",
        "bank_source_authority_review_path",
        "bank_post_generation_seal_path",
        "bank_post_generation_seal_review_path",
        "bank_generation_receipt_path",
        "bank_root",
    ):
        canonical_relative_path(document[key], key)
    _require_firewall(document)
    return document


def validate_authority_chain_documents(
    anchor: dict[str, Any],
    anchor_review: dict[str, Any],
    capability: dict[str, Any],
    release: dict[str, Any],
    *,
    expected_package: dict[str, str] | None = None,
) -> dict[str, str]:
    """Pure acyclic construction check used by runtime and static fixtures."""
    checked_anchor = validate_anchor_object(anchor, expected_package=expected_package)
    anchor_sha = bytes_sha256(canonical_json_bytes(anchor))
    validate_anchor_review_object(
        anchor_review,
        anchor_sha256=anchor_sha,
        package_bindings=checked_anchor["package_bindings"],
    )
    review_sha = bytes_sha256(canonical_json_bytes(anchor_review))
    validate_capability_object(
        capability,
        anchor_sha256=anchor_sha,
        bank_bindings=checked_anchor["bank_bindings"],
    )
    capability_sha = bytes_sha256(canonical_json_bytes(capability))
    validate_release_object(
        release,
        anchor_sha256=anchor_sha,
        anchor_review_sha256=review_sha,
        capability_sha256=capability_sha,
    )
    if "release_sha256" in capability:
        raise AuthorityError("capability must never bind the final release hash")
    return {
        "execution_anchor_sha256": anchor_sha,
        "execution_anchor_review_sha256": review_sha,
        "single_use_target_capability_sha256": capability_sha,
        "final_release_sha256": bytes_sha256(canonical_json_bytes(release)),
    }


def _expected_package_bindings() -> tuple[dict[str, str], dict[str, int]]:
    files = {
        "preregistration_sha256": secure_file(PREREG),
        "runner_sha256": secure_file(Path(__file__)),
        "static_verification_sha256": secure_file(STATIC),
        "post_generation_sealer_sha256": secure_file(POST_GENERATION_SEALER),
        "v1_hold_receipt_sha256": secure_file(V1_HOLD),
        "bank_source_authority_contract_sha256": secure_file(
            GENERATION_AUTHORITY_CONTRACT
        ),
        "post_generation_seal_contract_sha256": secure_file(
            POST_GENERATION_SEAL_CONTRACT
        ),
    }
    hashes = {name: hash_identity(identity) for name, identity in files.items()}
    # The V1 science runner is a future dynamic import.  Its reviewed digest is
    # authority metadata pre-claim; its actual bytes are first opened and hashed
    # by audited_full_preflight after capability consumption.
    hashes["v1_science_runner_sha256"] = V1_REVIEWED_SHA256["science_runner"]
    hashes["v2_runner_sha256"] = V2_REVIEWED_SHA256["runner"]
    hashes["v2_manifest_sha256"] = V2_REVIEWED_SHA256["manifest"]
    hashes["v2_independent_review_sha256"] = V2_REVIEWED_SHA256[
        "independent_review"
    ]
    if hashes["v1_hold_receipt_sha256"] != V1_REVIEWED_SHA256["hold_receipt"]:
        raise AuthorityError("V1 durable HOLD receipt changed")
    return hashes, {name: identity.size for name, identity in files.items()}


def validate_execution_authorities(
    release_path: Path, capability_path: Path
) -> ExecutionAuthority:
    release_identity = secure_file(release_path)
    release_raw, release = read_canonical_json(release_identity, "final release")
    exact_keys(release, RELEASE_KEYS, "final execution release")
    release_root = secure_directory(release_identity.unresolved.parent)

    anchor_identity = secure_relative_file(
        release_root, release.get("execution_anchor_path")
    )
    anchor_raw, anchor = read_canonical_json(anchor_identity, "execution anchor")
    anchor_review_identity = secure_relative_file(
        release_root, release.get("execution_anchor_review_path")
    )
    anchor_review_raw, anchor_review = read_canonical_json(
        anchor_review_identity, "execution anchor review"
    )
    expected_capability_identity = secure_relative_file(
        release_root, release.get("single_use_target_capability_path")
    )
    supplied_capability_identity = secure_file(capability_path)
    if (
        supplied_capability_identity.device != expected_capability_identity.device
        or supplied_capability_identity.inode != expected_capability_identity.inode
    ):
        raise AuthorityError("CLI capability differs from exact released capability")
    capability_raw, capability = read_canonical_json(
        supplied_capability_identity, "target capability"
    )

    expected_package, static_sizes = _expected_package_bindings()
    chain = validate_authority_chain_documents(
        anchor,
        anchor_review,
        capability,
        release,
        expected_package=expected_package,
    )
    if bytes_sha256(anchor_raw) != chain["execution_anchor_sha256"]:
        raise AuthorityError("execution anchor bytes are not canonical chain bytes")
    if bytes_sha256(anchor_review_raw) != chain["execution_anchor_review_sha256"]:
        raise AuthorityError("execution anchor review bytes changed")
    if bytes_sha256(capability_raw) != chain["single_use_target_capability_sha256"]:
        raise AuthorityError("target capability bytes changed")
    if bytes_sha256(release_raw) != chain["final_release_sha256"]:
        raise AuthorityError("final release bytes changed")

    external_files = {
        "bank_source_authority": secure_relative_file(
            release_root, release["bank_source_authority_path"]
        ),
        "bank_source_authority_review": secure_relative_file(
            release_root, release["bank_source_authority_review_path"]
        ),
        "bank_post_generation_seal": secure_relative_file(
            release_root, release["bank_post_generation_seal_path"]
        ),
        "bank_post_generation_seal_review": secure_relative_file(
            release_root, release["bank_post_generation_seal_review_path"]
        ),
        "bank_generation_receipt": secure_relative_file(
            release_root, release["bank_generation_receipt_path"]
        ),
    }
    bank_root = secure_relative_directory(release_root, release["bank_root"])
    preclaim_access = {
        "release_bytes_opened": len(release_raw),
        "anchor_bytes_opened": len(anchor_raw),
        "anchor_review_bytes_opened": len(anchor_review_raw),
        "capability_bytes_opened": len(capability_raw),
        "static_package_bytes_hashed": static_sizes,
        "manifest_files_opened_or_hashed": [],
        "dynamic_source_files_opened_or_hashed": [],
        "bank_bytes_opened": 0,
        "row_or_weight_bytes_opened": 0,
        "target_bundle_bytes_opened": 0,
        "target_values_decoded": 0,
    }
    return ExecutionAuthority(
        release_identity=release_identity,
        release_raw=release_raw,
        release=release,
        anchor_identity=anchor_identity,
        anchor_raw=anchor_raw,
        anchor=anchor,
        anchor_review_identity=anchor_review_identity,
        capability_identity=supplied_capability_identity,
        capability_raw=capability_raw,
        capability=capability,
        external_files=external_files,
        bank_root=bank_root,
        preclaim_access=preclaim_access,
    )


def consume_capability_unique(
    identity: PathIdentity, raw: bytes, capability_id: str
) -> PathIdentity:
    """Atomically rename one capability inode by hard-link then unlink.

    The transient second link is the mechanism, not a defect: after the
    directory is synced, unlinking the released name leaves exactly the same
    inode under the consumed name with one link.  No copied inode or O_EXCL
    capability-copy claim is made.
    """
    current = secure_file(identity.unresolved)
    if (
        current.device != identity.device
        or current.inode != identity.inode
        or current.size != identity.size
        or current.links != 1
    ):
        raise AuthorityError("capability identity or unique-link law changed")
    canonical_id(capability_id, "capability_id")
    consumed = identity.unresolved.with_name(
        f"{identity.unresolved.name}.consumed.{capability_id}.{bytes_sha256(raw)[:16]}"
    )
    if consumed.exists() or consumed.is_symlink():
        raise AuthorityError("consumed capability path already exists")
    os.link(identity.unresolved, consumed, follow_symlinks=False)
    linked = _lstat_without_symlink_components(consumed)
    if (
        not stat.S_ISREG(linked.st_mode)
        or linked.st_dev != identity.device
        or linked.st_ino != identity.inode
        or linked.st_nlink != 2
    ):
        raise AuthorityError("capability hard-link did not preserve exact inode")
    directory = os.open(identity.unresolved.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    before_unlink = _lstat_without_symlink_components(identity.unresolved)
    if (
        not stat.S_ISREG(before_unlink.st_mode)
        or before_unlink.st_dev != identity.device
        or before_unlink.st_ino != identity.inode
        or before_unlink.st_size != len(raw)
        or before_unlink.st_nlink != 2
    ):
        raise AuthorityError("capability swapped before unlink")
    os.unlink(identity.unresolved)
    directory = os.open(identity.unresolved.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if identity.unresolved.exists() or identity.unresolved.is_symlink():
        raise AuthorityError("original capability still exists after consumption")
    final_consumed = secure_file(consumed, expected_size=len(raw))
    if (
        final_consumed.links != 1
        or final_consumed.device != identity.device
        or final_consumed.inode != identity.inode
        or read_identity_bytes(final_consumed) != raw
    ):
        raise AuthorityError("consumed capability is not the original one-use inode")
    return final_consumed


SOURCE_AUTHORITY_KEYS = {
    "schema",
    "status",
    "authority_id",
    "issued_by",
    "issued_utc",
    "frozen_before_generation",
    "source_bindings",
    "row_scope_bindings",
    "permissions",
}
SOURCE_BINDING_KEYS = {
    "signed_pair_v2_manifest_sha256",
    "signed_pair_v2_independent_review_sha256",
    "three_bank_source_manifest_sha256",
    "three_bank_source_freeze_verification_sha256",
    "three_bank_source_independent_review_sha256",
    "generation_execution_anchor_sha256",
    "generation_runner_sha256",
    "literal_source_sha256",
    "capture_adapter_sha256",
    "generation_single_use_capability_sha256",
}
ROW_SCOPE_BINDING_KEYS = {
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
    "ordered_ranks_int32_sha256",
    "row_count",
}
SOURCE_PERMISSION_KEYS = {
    "source_import_allowed",
    "row_access_allowed",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "network_allowed",
    "package_allowed",
    "submission_allowed",
    "one_shot",
    "retry",
}
SOURCE_AUTHORITY_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "authority_sha256",
    "reviewed_before_generation",
    "target_free_scope_verified",
    "exact_source_allowlist_verified",
    "generation_authorized",
}
LEAN_SOURCE_AUTHORITY_KEYS = SOURCE_AUTHORITY_KEYS | {"generation_law"}
LEAN_SOURCE_BINDING_KEYS = {
    "signed_pair_v2_manifest_sha256",
    "signed_pair_v2_independent_review_sha256",
    "three_bank_source_manifest_sha256",
    "three_bank_source_freeze_verification_sha256",
    "three_bank_source_independent_review_sha256",
    "lean_runtime_package_manifest_sha256",
    "lean_runtime_freeze_receipt_sha256",
    "lean_runtime_freeze_verification_sha256",
    "lean_runtime_static_verification_sha256",
    "lean_runtime_independent_review_sha256",
    "generation_execution_anchor_sha256",
    "generation_single_use_capability_sha256",
    "generation_final_release_sha256",
    "generation_runner_sha256",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "runtime_token_sha256",
    "parent_capture_sha256",
    "literal_source_sha256",
    "independent_reference_source_sha256",
    "capture_adapter_sha256",
    "supersession_record_sha256",
}
LEAN_ROW_SCOPE_BINDING_KEYS = {
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
    "ordered_ranks_int32le_sha256",
    "row_count",
    "sentinel_ranks",
    "sentinel_ranks_int32le_sha256",
}
LEAN_SOURCE_AUTHORITY_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "authority_sha256",
    "supersession_record_sha256",
    "reviewed_before_generation",
    "target_free_scope_verified",
    "exact_source_allowlist_verified",
    "exact_generation_law_verified",
    "sentinel_contract_verified",
    "old_v2_capability_unconsumed_and_superseded",
    "transaction_root_empty",
    "generation_authorized",
}
V4_SOURCE_AUTHORITY_KEYS = {
    "schema",
    "status",
    "authority_id",
    "issued_by",
    "issued_utc",
    "frozen_before_generation",
    "source_bindings",
    "row_scope_bindings",
    "permissions",
    "manifest_compatibility_contract",
    "generation_law",
}
V4_SOURCE_BINDING_KEYS = {
    "signed_pair_v2_manifest_sha256",
    "signed_pair_v2_independent_review_sha256",
    "three_bank_source_manifest_sha256",
    "three_bank_source_freeze_verification_sha256",
    "three_bank_source_independent_review_sha256",
    "v4_runtime_package_manifest_sha256",
    "v4_runtime_freeze_receipt_sha256",
    "v4_runtime_freeze_verification_sha256",
    "v4_runtime_static_verification_sha256",
    "v4_runtime_independent_review_sha256",
    "generation_execution_anchor_sha256",
    "generation_single_use_capability_sha256",
    "generation_final_release_sha256",
    "generation_runner_sha256",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "runtime_token_sha256",
    "parent_capture_sha256",
    "literal_source_sha256",
    "independent_reference_source_sha256",
    "capture_adapter_sha256",
    "v3_terminal_failure_lineage_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "v3_exclusive_claim_sha256",
    "v3_terminal_failure_sha256",
    "v3_consumed_capability_sha256",
    "v3_terminal_successor_record_sha256",
    "prior_v2_authority_anchor_sha256",
    "prior_v2_single_use_capability_sha256",
    "prior_v2_final_release_sha256",
    "prior_v2_source_authority_sha256",
    "prior_v2_live_source_review_sha256",
}
V4_SOURCE_AUTHORITY_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "authority_sha256",
    "v3_terminal_successor_record_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "reviewed_before_generation",
    "target_free_scope_verified",
    "exact_source_allowlist_verified",
    "exact_generation_law_verified",
    "sentinel_contract_verified",
    "manifest_compatibility_scan_verified",
    "v3_terminal_claim_failure_and_consumption_verified",
    "v3_authority_or_transaction_reused",
    "prior_v2_capability_unconsumed_and_transaction_empty",
    "prior_v2_authority_or_transaction_reused",
    "fresh_v4_transaction_root_empty",
    "generation_authorized",
}
V5_SOURCE_AUTHORITY_KEYS = set(V4_SOURCE_AUTHORITY_KEYS)
V5_SOURCE_BINDING_KEYS = {
    "authority_builder_bindings_sha256",
    "authority_builder_freeze_receipt_sha256",
    "authority_builder_freeze_verification_sha256",
    "authority_builder_helper_sha256",
    "authority_builder_independent_review_sha256",
    "authority_builder_package_manifest_sha256",
    "authority_builder_static_verification_sha256",
    "signed_pair_v2_manifest_sha256",
    "signed_pair_v2_independent_review_sha256",
    "three_bank_source_manifest_sha256",
    "three_bank_source_freeze_verification_sha256",
    "three_bank_source_independent_review_sha256",
    "v5_runtime_package_manifest_sha256",
    "v5_runtime_freeze_receipt_sha256",
    "v5_runtime_freeze_verification_sha256",
    "v5_runtime_static_verification_sha256",
    "v5_runtime_independent_review_sha256",
    "flopscope_reset_verification_sha256",
    "generation_execution_anchor_sha256",
    "generation_single_use_capability_sha256",
    "generation_final_release_sha256",
    "generation_runner_sha256",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "runtime_token_sha256",
    "parent_capture_sha256",
    "literal_source_sha256",
    "independent_reference_source_sha256",
    "capture_adapter_sha256",
    "v3_terminal_failure_lineage_sha256",
    "v4_terminal_failure_lineage_sha256",
    "independent_incident_root_cause_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "v4_exclusive_claim_sha256",
    "v4_terminal_failure_sha256",
    "v4_consumed_capability_sha256",
    "v4_terminal_successor_record_sha256",
}
V5_SOURCE_AUTHORITY_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "authority_sha256",
    "authority_builder_independent_review_sha256",
    "v4_terminal_successor_record_sha256",
    "independent_incident_root_cause_sha256",
    "flopscope_reset_verification_sha256",
    "reviewed_before_generation",
    "target_free_scope_verified",
    "exact_source_allowlist_verified",
    "exact_generation_law_verified",
    "sentinel_contract_verified",
    "manifest_compatibility_scan_verified",
    "flopscope_post_commit_reset_law_verified",
    "v4_terminal_claim_failure_consumption_and_reset_verified",
    "v4_staging_forensic_only_and_not_input",
    "fresh_v5_transaction_root_empty",
    "generation_authorized",
}
LEAN_ORDERED_SCOPE_BINDINGS = {
    "ordered_688_row_scope_sha256": ORDERED_SCOPE_BINDINGS[
        "ordered_688_row_scope_sha256"
    ],
    "ordered_688_record_sha256": ORDERED_SCOPE_BINDINGS[
        "ordered_688_record_sha256"
    ],
    "ordered_ranks_int32le_sha256": ORDERED_SCOPE_BINDINGS[
        "ordered_ranks_int32_sha256"
    ],
    "row_count": ROWS,
    "sentinel_ranks": [0, 343, 687],
    "sentinel_ranks_int32le_sha256": (
        "9605c6e3790102dbb4418fa765c420f39e4300ff6983159dfe0345cbcc84c86a"
    ),
}
POST_SEAL_KEYS = {
    "schema",
    "status",
    "seal_id",
    "issued_by",
    "issued_utc",
    "pre_generation_source_authority_sha256",
    "pre_generation_source_authority_review_sha256",
    "runtime_v2_static_bindings",
    "generation_transaction_bindings",
    "ordered_scope_bindings",
    "bank_bindings",
    "transaction",
    "verification_evidence",
    "access_accounting",
    "limitations",
}
RUNTIME_STATIC_BINDING_KEYS = set(RUNTIME_V2_HASHES)
GENERATION_TRANSACTION_BINDING_KEYS = {
    "generation_claim_sha256",
    "generation_execution_anchor_sha256",
    "generation_final_release_sha256",
    "generation_single_use_capability_sha256",
    "consumed_capability_sha256",
    "generation_receipt_sha256",
}
SEAL_BANK_BINDING_KEYS = {
    "complete_bank_sha256",
    "A15_zero_sha256",
}
TRANSACTION_KEYS = {
    "claim_unique",
    "capability_input_had_one_link",
    "capability_consumed_by_same_inode_hard_link_then_unlink",
    "capability_copy_created_with_O_EXCL",
    "capability_inode_preserved",
    "capability_original_absent",
    "one_use",
    "retry",
}
SEAL_VERIFICATION_KEYS = {
    "generation_receipt_schema_exact_v2",
    "generation_receipt_canonical_bytes",
    "all_six_bank_files_hashed",
    "all_A15_bytes_zero",
    "sector_block_prefix_law_verified",
    "runtime_receipt_declares_target_accessed_false",
}
SEAL_ACCESS_KEYS = {
    "sealer_target_artifacts_opened",
    "sealer_target_values_decoded",
    "sealer_observer_selection_artifacts_opened",
    "network",
    "package",
    "submission",
}
SEAL_LIMITATION_KEYS = {
    "generation_target_freedom_basis",
    "hostile_interpreter_security_claim",
    "review_independence_cryptographically_proven",
}
POST_SEAL_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "post_generation_seal_sha256",
    "pre_generation_source_authority_sha256",
    "pre_generation_source_authority_review_sha256",
    "runtime_v2_static_bindings",
    "generation_transaction_bindings",
    "ordered_scope_bindings",
    "bank_bindings",
    "reviewed_after_generation",
    "receipt_and_all_six_bank_files_independently_rehashed",
    "sector_block_prefix_law_independently_replayed",
    "sealer_access_accounting_verified",
    "runtime_target_free_declaration_authenticated_not_observed",
    "generation_provenance_authenticated",
    "reviewer_different_from_seal_issuer",
    "hostile_interpreter_security_claim",
}
LEAN_POST_SEAL_KEYS = {
    "schema",
    "status",
    "seal_id",
    "issued_by",
    "issued_utc",
    "pre_generation_source_authority_sha256",
    "pre_generation_source_authority_review_sha256",
    "pre_generation_v3_terminal_successor_record_sha256",
    "v3_terminal_failure_bindings",
    "manifest_compatibility_scan_receipt_sha256",
    "runtime_v4_static_bindings",
    "generation_transaction_bindings",
    "ordered_scope_bindings",
    "lean_generation_law",
    "bank_bindings",
    "transaction",
    "verification_evidence",
    "access_accounting",
    "limitations",
}
LEAN_SEAL_VERIFICATION_KEYS = {
    "generation_receipt_schema_exact_v4",
    "generation_receipt_canonical_bytes",
    "all_six_bank_files_hashed",
    "all_A15_bytes_zero",
    "sector_block_prefix_law_verified",
    "sentinel_ranks_exact_0_343_687",
    "sentinel_parity_scope_authenticated_not_reexecuted",
    "all_row_dual_parity_claimed",
    "manifest_compatibility_scan_authenticated",
    "v3_terminal_failure_and_no_retry_authenticated",
    "runtime_receipt_declares_target_accessed_false",
}
LEAN_SEAL_LIMITATION_KEYS = {
    "generation_target_freedom_basis",
    "sentinel_parity_basis",
    "all_row_dual_parity_claimed",
    "hostile_interpreter_security_claim",
    "review_independence_cryptographically_proven",
}
LEAN_POST_SEAL_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "post_generation_seal_sha256",
    "pre_generation_source_authority_sha256",
    "pre_generation_source_authority_review_sha256",
    "pre_generation_v3_terminal_successor_record_sha256",
    "v3_terminal_failure_bindings",
    "manifest_compatibility_scan_receipt_sha256",
    "runtime_v4_static_bindings",
    "generation_transaction_bindings",
    "ordered_scope_bindings",
    "lean_generation_law",
    "bank_bindings",
    "reviewed_after_generation",
    "receipt_and_all_six_bank_files_independently_rehashed",
    "sector_block_prefix_law_independently_replayed",
    "sentinel_scope_and_no_all_row_overclaim_verified",
    "manifest_compatibility_scan_authenticated",
    "v3_terminal_failure_and_no_retry_authenticated",
    "sealer_access_accounting_verified",
    "runtime_target_free_declaration_authenticated_not_observed",
    "sentinel_parity_authenticated_not_reexecuted",
    "generation_provenance_authenticated",
    "reviewer_different_from_seal_issuer",
    "hostile_interpreter_security_claim",
}
V5_LEAN_POST_SEAL_KEYS = {
    "schema",
    "status",
    "seal_id",
    "issued_by",
    "issued_utc",
    "pre_generation_source_authority_sha256",
    "pre_generation_source_authority_review_sha256",
    "pre_generation_v4_terminal_successor_record_sha256",
    "v4_terminal_failure_bindings",
    "independent_incident_root_cause_sha256",
    "flopscope_reset_verification_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "runtime_v5_static_bindings",
    "generation_transaction_bindings",
    "ordered_scope_bindings",
    "lean_generation_law",
    "bank_bindings",
    "transaction",
    "verification_evidence",
    "access_accounting",
    "limitations",
}
V5_LEAN_SEAL_VERIFICATION_KEYS = {
    "generation_receipt_schema_exact_v4",
    "generation_receipt_and_bank_keysets_exact_v4",
    "generation_receipt_canonical_bytes",
    "all_six_bank_files_hashed",
    "all_A15_bytes_zero",
    "sector_block_prefix_law_verified",
    "sentinel_ranks_exact_0_343_687",
    "sentinel_parity_scope_authenticated_not_reexecuted",
    "all_row_dual_parity_claimed",
    "manifest_compatibility_scan_authenticated",
    "v4_terminal_failure_and_no_retry_authenticated",
    "independent_incident_root_cause_authenticated",
    "flopscope_post_commit_reset_law_authenticated",
    "runtime_receipt_declares_target_accessed_false",
}
V5_LEAN_SEAL_LIMITATION_KEYS = {
    "generation_target_freedom_basis",
    "sentinel_parity_basis",
    "flopscope_reset_basis",
    "flopscope_reset_execution_independently_observed",
    "all_row_dual_parity_claimed",
    "hostile_interpreter_security_claim",
    "review_independence_cryptographically_proven",
}
V5_LEAN_POST_SEAL_REVIEW_KEYS = {
    "schema",
    "status",
    "review_id",
    "issued_by",
    "issued_utc",
    "post_generation_seal_sha256",
    "pre_generation_source_authority_sha256",
    "pre_generation_source_authority_review_sha256",
    "pre_generation_v4_terminal_successor_record_sha256",
    "v4_terminal_failure_bindings",
    "independent_incident_root_cause_sha256",
    "flopscope_reset_verification_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "runtime_v5_static_bindings",
    "generation_transaction_bindings",
    "ordered_scope_bindings",
    "lean_generation_law",
    "bank_bindings",
    "reviewed_after_generation",
    "receipt_and_all_six_bank_files_independently_rehashed",
    "sector_block_prefix_law_independently_replayed",
    "sentinel_scope_and_no_all_row_overclaim_verified",
    "generation_receipt_and_bank_keysets_exact_v4_verified",
    "manifest_compatibility_scan_authenticated",
    "v4_terminal_failure_and_no_retry_authenticated",
    "independent_incident_root_cause_authenticated",
    "flopscope_post_commit_reset_law_authenticated_not_reexecuted",
    "sealer_access_accounting_verified",
    "runtime_target_free_declaration_authenticated_not_observed",
    "sentinel_parity_authenticated_not_reexecuted",
    "generation_provenance_authenticated",
    "reviewer_different_from_seal_issuer",
    "hostile_interpreter_security_claim",
}


def validate_source_authority_object(value: Any) -> dict[str, Any]:
    authority = exact_keys(value, SOURCE_AUTHORITY_KEYS, "pre-generation source authority")
    if (
        authority["schema"]
        != "public-q24-signed-pair-bank-source-authority-v2"
        or authority["status"]
        != "FROZEN_REVIEWED_ALLOWLIST_BEFORE_THREE_BANK_GENERATION"
        or authority["frozen_before_generation"] is not True
    ):
        raise AuthorityError("pre-generation source authority schema/status mismatch")
    canonical_id(authority["authority_id"], "source authority_id")
    canonical_id(authority["issued_by"], "source issued_by", exact="root")
    canonical_utc(authority["issued_utc"], "source issued_utc")
    source = exact_keys(
        authority["source_bindings"], SOURCE_BINDING_KEYS, "source allowlist bindings"
    )
    if any(not is_sha(item) for item in source.values()):
        raise AuthorityError("source allowlist contains invalid SHA256")
    if (
        source["signed_pair_v2_manifest_sha256"]
        != PREDECESSOR_HASHES[ACTION_V2 / "PACKAGE_MANIFEST.json"]
        or source["signed_pair_v2_independent_review_sha256"]
        != PREDECESSOR_HASHES[ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"]
        or source["three_bank_source_manifest_sha256"]
        != SOURCE_V1_HASHES["three_bank_source_manifest_sha256"]
        or source["three_bank_source_freeze_verification_sha256"]
        != SOURCE_V1_HASHES["three_bank_source_freeze_verification_sha256"]
        or source["three_bank_source_independent_review_sha256"]
        != SOURCE_V1_HASHES["three_bank_source_independent_review_sha256"]
        or source["generation_runner_sha256"]
        != RUNTIME_V2_HASHES["runner_sha256"]
        or source["literal_source_sha256"]
        != RUNTIME_V2_COMPONENT_SHA256["signed_pair_literal.py"]
        or source["capture_adapter_sha256"]
        != RUNTIME_V2_COMPONENT_SHA256["capture_adapter.py"]
    ):
        raise AuthorityError("source authority predecessor changed")
    scope = exact_keys(
        authority["row_scope_bindings"], ROW_SCOPE_BINDING_KEYS, "row scope bindings"
    )
    for key in ROW_SCOPE_BINDING_KEYS - {"row_count"}:
        if not is_sha(scope[key]):
            raise AuthorityError(f"row scope SHA invalid: {key}")
    if scope != ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("source authority ordered-scope binding changed")
    permissions = exact_keys(
        authority["permissions"], SOURCE_PERMISSION_KEYS, "source permissions"
    )
    expected_permissions = {
        "source_import_allowed": True,
        "row_access_allowed": True,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "network_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
        "one_shot": True,
        "retry": False,
    }
    if any(permissions[key] is not expected for key, expected in expected_permissions.items()):
        raise AuthorityError("source authority permission changed")
    return authority


def validate_source_authority_review_object(
    value: Any, *, authority_sha256: str
) -> dict[str, Any]:
    review = exact_keys(
        value, SOURCE_AUTHORITY_REVIEW_KEYS, "pre-generation authority review"
    )
    if (
        review["schema"]
        != "public-q24-signed-pair-bank-source-authority-independent-review-v2"
        or review["status"] != "GO_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
        or review["authority_sha256"] != authority_sha256
    ):
        raise AuthorityError("pre-generation authority review mismatch")
    canonical_id(review["review_id"], "source review_id")
    canonical_id(review["issued_by"], "source review issued_by", exact="independent_reviewer")
    canonical_utc(review["issued_utc"], "source review issued_utc")
    for key in (
        "reviewed_before_generation",
        "target_free_scope_verified",
        "exact_source_allowlist_verified",
        "generation_authorized",
    ):
        if review[key] is not True:
            raise AuthorityError(f"pre-generation authority review boolean mismatch: {key}")
    return review


def expected_lean_generation_law() -> dict[str, Any]:
    return {
        "all_688_rows_captured_d4_parent_once": True,
        "all_688_rows_literal_h256_once": True,
        "host_prefix_derivation": {
            "sector_count": 8,
            "sector_major": True,
            "H256_receivers_per_sector": 32,
            "H64_receivers_per_sector": 8,
            "H16_receivers_per_sector": 2,
            "exact_final_float32_byte_copy": True,
        },
        "sentinel_checks": {
            "ranks": list(SENTINEL_RANKS),
            "selection_target_independent": True,
            "all_five_parent_components_base_vs_capture_byte_equal": True,
            "literal_vs_independent_reference_h16_h64_h256_and_a15_byte_equal": True,
            "derived_h16_h64_vs_literal_byte_equal": True,
        },
        "all_row_base_capture_parity_claimed": False,
        "all_row_literal_reference_parity_claimed": False,
        "resource_observations_can_veto_generation": False,
        "bank_inventory": [
            "actions_H16.f32le",
            "actions_H64.f32le",
            "actions_H256.f32le",
            "A15_zero_H16.f32le",
            "A15_zero_H64.f32le",
            "A15_zero_H256.f32le",
            "THREE_BANK_GENERATION_RECEIPT.json",
        ],
    }


def expected_lean_v4_generation_law() -> dict[str, Any]:
    law = expected_lean_generation_law()
    law["manifest_compatibility_scan_required_before_capability"] = True
    law["v3_terminal_attempt_may_be_retried"] = False
    law["v4_is_a_new_distinct_attempt"] = True
    return law


def expected_lean_v5_generation_law() -> dict[str, Any]:
    law = expected_lean_generation_law()
    law["flopscope_reset_law"] = {
        "call": "session['flops'].budget_reset()",
        "exact_count_after_successful_validated_rows": 688,
        "exact_count_after_failed_or_unvalidated_rows": 0,
        "placement": "AFTER_VALIDATE_APPEND_OPTIONAL_PROGRESS_BEFORE_NEXT_ROW",
        "inside_active_BudgetContext": False,
        "detached_row_result_and_output_bytes_invariant": True,
        "reset_failure_fails_closed": True,
    }
    law["manifest_compatibility_scan_required_before_capability"] = True
    law["v3_terminal_attempt_may_be_retried"] = False
    law["v4_terminal_attempt_may_be_retried"] = False
    law["v4_staging_is_forensic_only_and_not_an_input"] = True
    law["v5_is_a_new_distinct_attempt"] = True
    return law


def expected_manifest_compatibility_contract() -> dict[str, Any]:
    return {
        "receipt_schema": "public-q24-row-manifest-compatibility-scan-receipt-v4",
        "receipt_status": "PASS_EXACT_REAL_SCHEMA_TARGET_BLIND_METADATA_ONLY",
        "independent_runtime_review_must_authenticate_receipt": True,
        "decoded_columns": ["mlp_id", "mlp_name", "mlp_seed", "weights"],
        "exact_python_int_zero_fields": [
            "scores",
            "target_columns_projected",
            "target_values_decoded",
            "fits",
            "optimizers_constructed",
        ],
        "exact_single_manifest_schema_required": True,
        "exact_single_manifest_keyset_required": True,
        "row_manifest_hashes_must_match_ordered_scope": True,
        "weights_opened": False,
        "targets_opened": False,
        "numerical_module_imported": False,
    }


def validate_lean_v4_source_authority_object(value: Any) -> dict[str, Any]:
    authority = exact_keys(
        value, V4_SOURCE_AUTHORITY_KEYS, "lean V4 pre-generation source authority"
    )
    if (
        authority["schema"]
        != "public-q24-signed-pair-lean-bank-source-authority-v4"
        or authority["status"]
        != "FROZEN_REVIEWED_V4_ALLOWLIST_BEFORE_BANK_GENERATION"
        or authority["frozen_before_generation"] is not True
    ):
        raise AuthorityError("lean V4 source authority schema/status mismatch")
    canonical_id(authority["authority_id"], "lean V4 source authority_id")
    canonical_id(authority["issued_by"], "lean V4 source issuer", exact="root")
    canonical_utc(authority["issued_utc"], "lean V4 source issued_utc")
    source = exact_keys(
        authority["source_bindings"],
        V4_SOURCE_BINDING_KEYS,
        "lean V4 source bindings",
    )
    if any(not is_sha(item) for item in source.values()):
        raise AuthorityError("lean V4 source binding is not SHA256")
    expected_lineage = {
        "signed_pair_v2_manifest_sha256": PREDECESSOR_HASHES[
            ACTION_V2 / "PACKAGE_MANIFEST.json"
        ],
        "signed_pair_v2_independent_review_sha256": PREDECESSOR_HASHES[
            ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"
        ],
        **SOURCE_V1_HASHES,
        **PRIOR_V2_NONREUSE_BINDINGS,
    }
    if any(source[key] != expected for key, expected in expected_lineage.items()):
        raise AuthorityError("lean V4 source lineage changed")
    expected_runtime = {
        "v4_runtime_package_manifest_sha256": RUNTIME_V4_HASHES[
            "package_manifest_sha256"
        ],
        "v4_runtime_freeze_receipt_sha256": RUNTIME_V4_HASHES[
            "freeze_receipt_sha256"
        ],
        "v4_runtime_freeze_verification_sha256": RUNTIME_V4_HASHES[
            "freeze_verification_sha256"
        ],
        "v4_runtime_static_verification_sha256": RUNTIME_V4_HASHES[
            "static_verification_sha256"
        ],
        "v4_runtime_independent_review_sha256": RUNTIME_V4_HASHES[
            "independent_review_sha256"
        ],
        "generation_runner_sha256": RUNTIME_V4_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": RUNTIME_V4_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": RUNTIME_V4_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "runtime_token_sha256": RUNTIME_V4_COMPONENT_SHA256["runtime_token.py"],
        "parent_capture_sha256": RUNTIME_V4_COMPONENT_SHA256["parent_capture.py"],
        "literal_source_sha256": RUNTIME_V4_COMPONENT_SHA256[
            "signed_pair_literal.py"
        ],
        "independent_reference_source_sha256": RUNTIME_V4_COMPONENT_SHA256[
            "signed_pair_reference.py"
        ],
        "capture_adapter_sha256": RUNTIME_V4_COMPONENT_SHA256[
            "capture_adapter.py"
        ],
        "v3_terminal_failure_lineage_sha256": RUNTIME_V4_HASHES[
            "v3_terminal_failure_lineage_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": RUNTIME_V4_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "v3_exclusive_claim_sha256": V3_TERMINAL_FAILURE_BINDINGS[
            "exclusive_claim_sha256"
        ],
        "v3_terminal_failure_sha256": V3_TERMINAL_FAILURE_BINDINGS[
            "terminal_failure_sha256"
        ],
        "v3_consumed_capability_sha256": V3_TERMINAL_FAILURE_BINDINGS[
            "consumed_capability_sha256"
        ],
        "v3_terminal_successor_record_sha256": V4_SUCCESSOR_RECORD_SHA256,
    }
    if any(source[key] != expected for key, expected in expected_runtime.items()):
        raise AuthorityError("lean V4 source runtime/terminal binding changed")
    if authority["row_scope_bindings"] != LEAN_ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("lean V4 source ordered/sentinel scope changed")
    if (
        authority["manifest_compatibility_contract"]
        != expected_manifest_compatibility_contract()
    ):
        raise AuthorityError("lean V4 manifest compatibility contract changed")
    if authority["generation_law"] != expected_lean_v4_generation_law():
        raise AuthorityError("lean V4 generation law changed")
    permissions = exact_keys(
        authority["permissions"], SOURCE_PERMISSION_KEYS, "lean V4 permissions"
    )
    expected_permissions = {
        "source_import_allowed": True,
        "row_access_allowed": True,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "network_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
        "one_shot": True,
        "retry": False,
    }
    if any(
        permissions[key] is not expected
        for key, expected in expected_permissions.items()
    ):
        raise AuthorityError("lean V4 source permission changed")
    return authority


def validate_lean_v4_source_authority_review_object(
    value: Any, *, authority_sha256: str, authority: dict[str, Any]
) -> dict[str, Any]:
    review = exact_keys(
        value,
        V4_SOURCE_AUTHORITY_REVIEW_KEYS,
        "lean V4 pre-generation source authority review",
    )
    if (
        review["schema"]
        != "public-q24-signed-pair-lean-bank-source-authority-independent-review-v4"
        or review["status"]
        != "GO_LEAN_V4_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
        or review["authority_sha256"] != authority_sha256
        or review["v3_terminal_successor_record_sha256"]
        != V4_SUCCESSOR_RECORD_SHA256
        or review["manifest_compatibility_scan_receipt_sha256"]
        != RUNTIME_V4_HASHES["manifest_compatibility_scan_receipt_sha256"]
    ):
        raise AuthorityError("lean V4 source authority review mismatch")
    canonical_id(review["review_id"], "lean V4 source review_id")
    canonical_id(
        review["issued_by"],
        "lean V4 source review issuer",
        exact="independent_reviewer",
    )
    issued = canonical_utc(review["issued_utc"], "lean V4 source review issued_utc")
    if issued < authority["issued_utc"]:
        raise AuthorityError("lean V4 source review predates authority")
    for key in (
        "reviewed_before_generation",
        "target_free_scope_verified",
        "exact_source_allowlist_verified",
        "exact_generation_law_verified",
        "sentinel_contract_verified",
        "manifest_compatibility_scan_verified",
        "v3_terminal_claim_failure_and_consumption_verified",
        "prior_v2_capability_unconsumed_and_transaction_empty",
        "fresh_v4_transaction_root_empty",
        "generation_authorized",
    ):
        if review[key] is not True:
            raise AuthorityError(f"lean V4 source review boolean mismatch: {key}")
    if review["v3_authority_or_transaction_reused"] is not False:
        raise AuthorityError("lean V4 source review permits V3 transaction reuse")
    if review["prior_v2_authority_or_transaction_reused"] is not False:
        raise AuthorityError("lean V4 source review permits prior V2 transaction reuse")
    return review


def validate_lean_v5_source_authority_object(value: Any) -> dict[str, Any]:
    """Validate the fresh V5 pre-generation allowlist and V4 terminal closure."""
    if not all(
        is_sha(value)
        for value in (
            V5_SOURCE_AUTHORITY_SHA256,
            V5_SOURCE_AUTHORITY_REVIEW_SHA256,
            V5_SUCCESSOR_RECORD_SHA256,
        )
    ):
        raise AuthorityError("lean V5 live source roots are not frozen")
    authority = exact_keys(
        value, V5_SOURCE_AUTHORITY_KEYS, "lean V5 pre-generation source authority"
    )
    if (
        authority["schema"]
        != "public-q24-signed-pair-lean-bank-source-authority-v5"
        or authority["status"]
        != "FROZEN_REVIEWED_V5_ALLOWLIST_BEFORE_BANK_GENERATION"
        or authority["frozen_before_generation"] is not True
    ):
        raise AuthorityError("lean V5 source authority schema/status mismatch")
    canonical_id(authority["authority_id"], "lean V5 source authority_id")
    canonical_id(authority["issued_by"], "lean V5 source issuer", exact="root")
    canonical_utc(authority["issued_utc"], "lean V5 source issued_utc")
    source = exact_keys(
        authority["source_bindings"],
        V5_SOURCE_BINDING_KEYS,
        "lean V5 source bindings",
    )
    if any(not is_sha(item) for item in source.values()):
        raise AuthorityError("lean V5 source binding is not SHA256")
    expected_lineage = {
        "signed_pair_v2_manifest_sha256": PREDECESSOR_HASHES[
            ACTION_V2 / "PACKAGE_MANIFEST.json"
        ],
        "signed_pair_v2_independent_review_sha256": PREDECESSOR_HASHES[
            ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"
        ],
        **SOURCE_V1_HASHES,
    }
    if any(source[key] != expected for key, expected in expected_lineage.items()):
        raise AuthorityError("lean V5 accepted source lineage changed")
    expected_runtime = {
        **V5_AUTHORITY_BUILDER_HASHES,
        "v5_runtime_package_manifest_sha256": RUNTIME_V5_HASHES[
            "package_manifest_sha256"
        ],
        "v5_runtime_freeze_receipt_sha256": RUNTIME_V5_HASHES[
            "freeze_receipt_sha256"
        ],
        "v5_runtime_freeze_verification_sha256": RUNTIME_V5_HASHES[
            "freeze_verification_sha256"
        ],
        "v5_runtime_static_verification_sha256": RUNTIME_V5_HASHES[
            "static_verification_sha256"
        ],
        "v5_runtime_independent_review_sha256": RUNTIME_V5_HASHES[
            "independent_review_sha256"
        ],
        "flopscope_reset_verification_sha256": RUNTIME_V5_HASHES[
            "flopscope_reset_verification_sha256"
        ],
        "generation_runner_sha256": RUNTIME_V5_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": RUNTIME_V5_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": RUNTIME_V5_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "runtime_token_sha256": RUNTIME_V5_COMPONENT_SHA256["runtime_token.py"],
        "parent_capture_sha256": RUNTIME_V5_COMPONENT_SHA256["parent_capture.py"],
        "literal_source_sha256": RUNTIME_V5_COMPONENT_SHA256[
            "signed_pair_literal.py"
        ],
        "independent_reference_source_sha256": RUNTIME_V5_COMPONENT_SHA256[
            "signed_pair_reference.py"
        ],
        "capture_adapter_sha256": RUNTIME_V5_COMPONENT_SHA256[
            "capture_adapter.py"
        ],
        "v3_terminal_failure_lineage_sha256": RUNTIME_V5_HASHES[
            "v3_terminal_failure_lineage_sha256"
        ],
        "v4_terminal_failure_lineage_sha256": RUNTIME_V5_HASHES[
            "v4_terminal_failure_lineage_sha256"
        ],
        "independent_incident_root_cause_sha256": RUNTIME_V5_HASHES[
            "independent_incident_root_cause_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": RUNTIME_V5_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "v4_exclusive_claim_sha256": V4_TERMINAL_FAILURE_BINDINGS[
            "exclusive_claim_sha256"
        ],
        "v4_terminal_failure_sha256": V4_TERMINAL_FAILURE_BINDINGS[
            "terminal_failure_sha256"
        ],
        "v4_consumed_capability_sha256": V4_TERMINAL_FAILURE_BINDINGS[
            "consumed_capability_sha256"
        ],
        "v4_terminal_successor_record_sha256": V5_SUCCESSOR_RECORD_SHA256,
    }
    if any(source[key] != expected for key, expected in expected_runtime.items()):
        raise AuthorityError("lean V5 source runtime/terminal/reset binding changed")
    if authority["row_scope_bindings"] != LEAN_ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("lean V5 source ordered/sentinel scope changed")
    if (
        authority["manifest_compatibility_contract"]
        != expected_manifest_compatibility_contract()
    ):
        raise AuthorityError("lean V5 manifest compatibility contract changed")
    if authority["generation_law"] != expected_lean_v5_generation_law():
        raise AuthorityError("lean V5 generation/reset law changed")
    permissions = exact_keys(
        authority["permissions"], SOURCE_PERMISSION_KEYS, "lean V5 permissions"
    )
    expected_permissions = {
        "source_import_allowed": True,
        "row_access_allowed": True,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "network_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
        "one_shot": True,
        "retry": False,
    }
    if any(
        permissions[key] is not expected
        for key, expected in expected_permissions.items()
    ):
        raise AuthorityError("lean V5 source permission changed")
    return authority


def validate_lean_v5_source_authority_review_object(
    value: Any, *, authority_sha256: str, authority: dict[str, Any]
) -> dict[str, Any]:
    review = exact_keys(
        value,
        V5_SOURCE_AUTHORITY_REVIEW_KEYS,
        "lean V5 pre-generation source authority review",
    )
    if (
        review["schema"]
        != "public-q24-signed-pair-lean-bank-source-authority-independent-review-v5"
        or review["status"]
        != "GO_LEAN_V5_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
        or review["authority_sha256"] != authority_sha256
        or review["authority_builder_independent_review_sha256"]
        != V5_AUTHORITY_BUILDER_HASHES[
            "authority_builder_independent_review_sha256"
        ]
        or review["v4_terminal_successor_record_sha256"]
        != V5_SUCCESSOR_RECORD_SHA256
        or review["independent_incident_root_cause_sha256"]
        != RUNTIME_V5_HASHES["independent_incident_root_cause_sha256"]
        or review["flopscope_reset_verification_sha256"]
        != RUNTIME_V5_HASHES["flopscope_reset_verification_sha256"]
    ):
        raise AuthorityError("lean V5 source authority review mismatch")
    canonical_id(review["review_id"], "lean V5 source review_id")
    canonical_id(
        review["issued_by"],
        "lean V5 source review issuer",
        exact="independent_reviewer",
    )
    issued = canonical_utc(review["issued_utc"], "lean V5 source review issued_utc")
    if issued < authority["issued_utc"]:
        raise AuthorityError("lean V5 source review predates authority")
    for key in (
        "reviewed_before_generation",
        "target_free_scope_verified",
        "exact_source_allowlist_verified",
        "exact_generation_law_verified",
        "sentinel_contract_verified",
        "manifest_compatibility_scan_verified",
        "flopscope_post_commit_reset_law_verified",
        "v4_terminal_claim_failure_consumption_and_reset_verified",
        "v4_staging_forensic_only_and_not_input",
        "fresh_v5_transaction_root_empty",
        "generation_authorized",
    ):
        if review[key] is not True:
            raise AuthorityError(f"lean V5 source review boolean mismatch: {key}")
    return review


def validate_lean_source_authority_object(value: Any) -> dict[str, Any]:
    if set(RUNTIME_V3_HASHES) != set(RUNTIME_V2_HASHES) or set(
        RUNTIME_V3_COMPONENT_SHA256
    ) != set(RUNTIME_V2_COMPONENT_SHA256):
        raise AuthorityError("lean runtime-V3 roots are not frozen and complete")
    authority = exact_keys(
        value, LEAN_SOURCE_AUTHORITY_KEYS, "lean pre-generation source authority"
    )
    if (
        authority["schema"]
        != "public-q24-signed-pair-lean-bank-source-authority-v3"
        or authority["status"]
        != "FROZEN_REVIEWED_LEAN_ALLOWLIST_BEFORE_BANK_GENERATION"
        or authority["frozen_before_generation"] is not True
    ):
        raise AuthorityError("lean pre-generation source authority schema/status mismatch")
    canonical_id(authority["authority_id"], "lean source authority_id")
    canonical_id(authority["issued_by"], "lean source issued_by", exact="root")
    canonical_utc(authority["issued_utc"], "lean source issued_utc")
    source = exact_keys(
        authority["source_bindings"],
        LEAN_SOURCE_BINDING_KEYS,
        "lean source allowlist bindings",
    )
    if any(not is_sha(item) for item in source.values()):
        raise AuthorityError("lean source allowlist contains invalid SHA256")
    if (
        source["signed_pair_v2_manifest_sha256"]
        != PREDECESSOR_HASHES[ACTION_V2 / "PACKAGE_MANIFEST.json"]
        or source["signed_pair_v2_independent_review_sha256"]
        != PREDECESSOR_HASHES[ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"]
        or source["three_bank_source_manifest_sha256"]
        != SOURCE_V1_HASHES["three_bank_source_manifest_sha256"]
        or source["three_bank_source_freeze_verification_sha256"]
        != SOURCE_V1_HASHES["three_bank_source_freeze_verification_sha256"]
        or source["three_bank_source_independent_review_sha256"]
        != SOURCE_V1_HASHES["three_bank_source_independent_review_sha256"]
        or source["lean_runtime_package_manifest_sha256"]
        != RUNTIME_V3_HASHES["package_manifest_sha256"]
        or source["lean_runtime_freeze_receipt_sha256"]
        != RUNTIME_V3_HASHES["freeze_receipt_sha256"]
        or source["lean_runtime_freeze_verification_sha256"]
        != RUNTIME_V3_HASHES["freeze_verification_sha256"]
        or source["lean_runtime_static_verification_sha256"]
        != RUNTIME_V3_HASHES["static_verification_sha256"]
        or source["lean_runtime_independent_review_sha256"]
        != RUNTIME_V3_HASHES["independent_review_sha256"]
        or source["generation_runner_sha256"]
        != RUNTIME_V3_HASHES["runner_sha256"]
        or source["callable_bundle_manifest_sha256"]
        != RUNTIME_V3_HASHES["callable_bundle_manifest_sha256"]
        or source["execution_stack_manifest_sha256"]
        != RUNTIME_V3_HASHES["execution_stack_manifest_sha256"]
        or source["runtime_token_sha256"]
        != RUNTIME_V3_COMPONENT_SHA256["runtime_token.py"]
        or source["parent_capture_sha256"]
        != RUNTIME_V3_COMPONENT_SHA256["parent_capture.py"]
        or source["literal_source_sha256"]
        != RUNTIME_V3_COMPONENT_SHA256["signed_pair_literal.py"]
        or source["independent_reference_source_sha256"]
        != RUNTIME_V3_COMPONENT_SHA256["signed_pair_reference.py"]
        or source["capture_adapter_sha256"]
        != RUNTIME_V3_COMPONENT_SHA256["capture_adapter.py"]
        or source["supersession_record_sha256"]
        != LEAN_SUPERSESSION_RECORD_SHA256
    ):
        raise AuthorityError("lean source authority predecessor/root changed")
    scope = exact_keys(
        authority["row_scope_bindings"],
        LEAN_ROW_SCOPE_BINDING_KEYS,
        "lean row-scope bindings",
    )
    if scope != LEAN_ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("lean source authority ordered/sentinel scope changed")
    if authority["generation_law"] != expected_lean_generation_law():
        raise AuthorityError("lean source authority generation law changed")
    permissions = exact_keys(
        authority["permissions"], SOURCE_PERMISSION_KEYS, "lean source permissions"
    )
    expected_permissions = {
        "source_import_allowed": True,
        "row_access_allowed": True,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "network_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
        "one_shot": True,
        "retry": False,
    }
    if any(
        permissions[key] is not expected
        for key, expected in expected_permissions.items()
    ):
        raise AuthorityError("lean source authority permission changed")
    return authority


def validate_lean_source_authority_review_object(
    value: Any, *, authority_sha256: str, authority: dict[str, Any]
) -> dict[str, Any]:
    review = exact_keys(
        value,
        LEAN_SOURCE_AUTHORITY_REVIEW_KEYS,
        "lean pre-generation authority review",
    )
    if (
        review["schema"]
        != "public-q24-signed-pair-lean-bank-source-authority-independent-review-v3"
        or review["status"]
        != "GO_LEAN_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
        or review["authority_sha256"] != authority_sha256
        or review["supersession_record_sha256"]
        != authority["source_bindings"]["supersession_record_sha256"]
    ):
        raise AuthorityError("lean pre-generation authority review mismatch")
    canonical_id(review["review_id"], "lean source review_id")
    canonical_id(
        review["issued_by"],
        "lean source review issued_by",
        exact="independent_reviewer",
    )
    issued = canonical_utc(review["issued_utc"], "lean source review issued_utc")
    if issued < authority["issued_utc"]:
        raise AuthorityError("lean source review predates source authority")
    for key in (
        "reviewed_before_generation",
        "target_free_scope_verified",
        "exact_source_allowlist_verified",
        "exact_generation_law_verified",
        "sentinel_contract_verified",
        "old_v2_capability_unconsumed_and_superseded",
        "transaction_root_empty",
        "generation_authorized",
    ):
        if review[key] is not True:
            raise AuthorityError(f"lean source review boolean mismatch: {key}")
    return review


def validate_post_generation_seal_object(value: Any) -> dict[str, Any]:
    seal = exact_keys(value, POST_SEAL_KEYS, "post-generation seal")
    if (
        seal["schema"]
        != "public-q24-signed-pair-three-bank-post-generation-seal-v2"
        or seal["status"]
        != "SEALED_RUNTIME_V2_TARGET_BLIND_THREE_BANK_GENERATION"
    ):
        raise AuthorityError("post-generation seal schema/status mismatch")
    canonical_id(seal["seal_id"], "post-generation seal_id")
    canonical_id(
        seal["issued_by"],
        "post-generation issued_by",
        exact="post_generation_sealer",
    )
    canonical_utc(seal["issued_utc"], "post-generation issued_utc")
    if not is_sha(seal["pre_generation_source_authority_sha256"]) or not is_sha(
        seal["pre_generation_source_authority_review_sha256"]
    ):
        raise AuthorityError("post-generation pre-authority binding invalid")
    runtime = exact_keys(
        seal["runtime_v2_static_bindings"],
        RUNTIME_STATIC_BINDING_KEYS,
        "runtime-V2 static bindings",
    )
    if runtime != RUNTIME_V2_HASHES:
        raise AuthorityError("post-generation seal runtime-V2 roots changed")
    transaction_bindings = exact_keys(
        seal["generation_transaction_bindings"],
        GENERATION_TRANSACTION_BINDING_KEYS,
        "generation transaction bindings",
    )
    if any(not is_sha(item) for item in transaction_bindings.values()):
        raise AuthorityError("post-generation transaction binding invalid")
    if (
        transaction_bindings["generation_single_use_capability_sha256"]
        != transaction_bindings["consumed_capability_sha256"]
    ):
        raise AuthorityError("generation capability digest changed during consumption")
    scope = exact_keys(
        seal["ordered_scope_bindings"],
        ROW_SCOPE_BINDING_KEYS,
        "seal ordered-scope bindings",
    )
    if scope != ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("seal ordered-scope binding changed")
    banks = exact_keys(
        seal["bank_bindings"], SEAL_BANK_BINDING_KEYS, "seal bank bindings"
    )
    validate_sha_map(banks["complete_bank_sha256"], "seal complete bank hashes")
    validate_sha_map(banks["A15_zero_sha256"], "seal A15 hashes")
    transaction = exact_keys(seal["transaction"], TRANSACTION_KEYS, "generation transaction")
    expected_transaction = {
        "claim_unique": True,
        "capability_input_had_one_link": True,
        "capability_consumed_by_same_inode_hard_link_then_unlink": True,
        "capability_copy_created_with_O_EXCL": False,
        "capability_inode_preserved": True,
        "capability_original_absent": True,
        "one_use": True,
        "retry": False,
    }
    if any(transaction[key] is not expected for key, expected in expected_transaction.items()):
        raise AuthorityError("post-generation transaction assertion changed")
    evidence = exact_keys(
        seal["verification_evidence"],
        SEAL_VERIFICATION_KEYS,
        "post-generation verification evidence",
    )
    if any(evidence[key] is not True for key in SEAL_VERIFICATION_KEYS):
        raise AuthorityError("post-generation verification evidence incomplete")
    access = exact_keys(
        seal["access_accounting"], SEAL_ACCESS_KEYS, "sealer access accounting"
    )
    for key in (
        "sealer_target_artifacts_opened",
        "sealer_target_values_decoded",
        "sealer_observer_selection_artifacts_opened",
    ):
        if strict_int(access[key], minimum=0) != 0:
            raise AuthorityError(f"sealer target firewall changed: {key}")
    for key in ("network", "package", "submission"):
        if access[key] is not False:
            raise AuthorityError(f"sealer external-action firewall changed: {key}")
    limitations = exact_keys(
        seal["limitations"], SEAL_LIMITATION_KEYS, "post-generation seal limitations"
    )
    if (
        limitations["generation_target_freedom_basis"]
        != "FROZEN_RUNTIME_CONTROL_FLOW_PLUS_AUTHENTICATED_RUNTIME_RECEIPT_DECLARATION__NOT_EXTERNAL_PROCESS_OBSERVATION"
        or limitations["hostile_interpreter_security_claim"] is not False
        or limitations["review_independence_cryptographically_proven"] is not False
    ):
        raise AuthorityError("post-generation seal overclaims its evidence")
    return seal


def validate_post_generation_seal_review_object(
    value: Any, *, seal_sha256: str, seal: dict[str, Any]
) -> dict[str, Any]:
    review = exact_keys(value, POST_SEAL_REVIEW_KEYS, "post-generation seal review")
    if (
        review["schema"]
        != "public-q24-signed-pair-three-bank-post-generation-seal-independent-review-v2"
        or review["status"]
        != "GO_AUTHENTICATED_RUNTIME_V2_TARGET_BLIND_THREE_BANKS"
        or review["post_generation_seal_sha256"] != seal_sha256
        or review["pre_generation_source_authority_sha256"]
        != seal["pre_generation_source_authority_sha256"]
        or review["pre_generation_source_authority_review_sha256"]
        != seal["pre_generation_source_authority_review_sha256"]
        or review["runtime_v2_static_bindings"]
        != seal["runtime_v2_static_bindings"]
        or review["generation_transaction_bindings"]
        != seal["generation_transaction_bindings"]
        or review["ordered_scope_bindings"] != seal["ordered_scope_bindings"]
        or review["bank_bindings"] != seal["bank_bindings"]
    ):
        raise AuthorityError("post-generation independent review binding mismatch")
    canonical_id(review["review_id"], "post-generation review_id")
    canonical_id(review["issued_by"], "post-generation review issued_by", exact="independent_reviewer")
    issued = canonical_utc(review["issued_utc"], "post-generation review issued_utc")
    if issued < seal["issued_utc"]:
        raise AuthorityError("post-generation review predates seal")
    for key in (
        "reviewed_after_generation",
        "receipt_and_all_six_bank_files_independently_rehashed",
        "sector_block_prefix_law_independently_replayed",
        "sealer_access_accounting_verified",
        "runtime_target_free_declaration_authenticated_not_observed",
        "generation_provenance_authenticated",
        "reviewer_different_from_seal_issuer",
    ):
        if review[key] is not True:
            raise AuthorityError(f"post-generation review boolean mismatch: {key}")
    if review["hostile_interpreter_security_claim"] is not False:
        raise AuthorityError("post-generation review overclaims hostile-interpreter security")
    return review


def validate_lean_post_generation_seal_object(value: Any) -> dict[str, Any]:
    seal = exact_keys(value, LEAN_POST_SEAL_KEYS, "lean post-generation seal")
    if (
        seal["schema"]
        != "public-q24-signed-pair-lean-three-bank-post-generation-seal-v4"
        or seal["status"]
        != "SEALED_LEAN_RUNTIME_V4_TARGET_BLIND_THREE_BANK_GENERATION"
    ):
        raise AuthorityError("lean post-generation seal schema/status mismatch")
    canonical_id(seal["seal_id"], "lean post-generation seal_id")
    canonical_id(
        seal["issued_by"],
        "lean post-generation issued_by",
        exact="post_generation_sealer",
    )
    canonical_utc(seal["issued_utc"], "lean post-generation issued_utc")
    for key in (
        "pre_generation_source_authority_sha256",
        "pre_generation_source_authority_review_sha256",
        "pre_generation_v3_terminal_successor_record_sha256",
        "manifest_compatibility_scan_receipt_sha256",
    ):
        if not is_sha(seal[key]):
            raise AuthorityError(f"lean post-generation binding invalid: {key}")
    if (
        seal["pre_generation_v3_terminal_successor_record_sha256"]
        != V4_SUCCESSOR_RECORD_SHA256
        or seal["manifest_compatibility_scan_receipt_sha256"]
        != RUNTIME_V4_HASHES["manifest_compatibility_scan_receipt_sha256"]
    ):
        raise AuthorityError("lean post-generation V4 lineage binding changed")
    terminal = exact_keys(
        seal["v3_terminal_failure_bindings"],
        set(V3_TERMINAL_FAILURE_BINDINGS),
        "V3 terminal failure bindings",
    )
    if terminal != V3_TERMINAL_FAILURE_BINDINGS:
        raise AuthorityError("lean post-generation V3 terminal record changed")
    runtime = exact_keys(
        seal["runtime_v4_static_bindings"],
        set(RUNTIME_V4_HASHES),
        "lean runtime-V4 static bindings",
    )
    if runtime != RUNTIME_V4_HASHES:
        raise AuthorityError("lean post-generation seal runtime-V4 roots changed")
    transaction_bindings = exact_keys(
        seal["generation_transaction_bindings"],
        GENERATION_TRANSACTION_BINDING_KEYS,
        "lean generation transaction bindings",
    )
    if any(not is_sha(item) for item in transaction_bindings.values()):
        raise AuthorityError("lean post-generation transaction binding invalid")
    if (
        transaction_bindings["generation_single_use_capability_sha256"]
        != transaction_bindings["consumed_capability_sha256"]
    ):
        raise AuthorityError("lean generation capability digest changed during consumption")
    scope = exact_keys(
        seal["ordered_scope_bindings"],
        LEAN_ROW_SCOPE_BINDING_KEYS,
        "lean seal ordered/sentinel scope bindings",
    )
    if scope != LEAN_ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("lean seal ordered/sentinel scope changed")
    if seal["lean_generation_law"] != expected_lean_v4_generation_law():
        raise AuthorityError("lean seal generation law changed")
    banks = exact_keys(
        seal["bank_bindings"], SEAL_BANK_BINDING_KEYS, "lean seal bank bindings"
    )
    validate_sha_map(banks["complete_bank_sha256"], "lean seal complete bank hashes")
    validate_sha_map(banks["A15_zero_sha256"], "lean seal A15 hashes")
    transaction = exact_keys(
        seal["transaction"], TRANSACTION_KEYS, "lean generation transaction"
    )
    expected_transaction = {
        "claim_unique": True,
        "capability_input_had_one_link": True,
        "capability_consumed_by_same_inode_hard_link_then_unlink": True,
        "capability_copy_created_with_O_EXCL": False,
        "capability_inode_preserved": True,
        "capability_original_absent": True,
        "one_use": True,
        "retry": False,
    }
    if any(
        transaction[key] is not expected
        for key, expected in expected_transaction.items()
    ):
        raise AuthorityError("lean post-generation transaction assertion changed")
    evidence = exact_keys(
        seal["verification_evidence"],
        LEAN_SEAL_VERIFICATION_KEYS,
        "lean post-generation verification evidence",
    )
    for key in LEAN_SEAL_VERIFICATION_KEYS - {"all_row_dual_parity_claimed"}:
        if evidence[key] is not True:
            raise AuthorityError(f"lean verification evidence incomplete: {key}")
    if evidence["all_row_dual_parity_claimed"] is not False:
        raise AuthorityError("lean post-generation seal overclaims all-row parity")
    access = exact_keys(
        seal["access_accounting"], SEAL_ACCESS_KEYS, "lean sealer access accounting"
    )
    for key in (
        "sealer_target_artifacts_opened",
        "sealer_target_values_decoded",
        "sealer_observer_selection_artifacts_opened",
    ):
        if strict_int(access[key], minimum=0) != 0:
            raise AuthorityError(f"lean sealer target firewall changed: {key}")
    for key in ("network", "package", "submission"):
        if access[key] is not False:
            raise AuthorityError(f"lean sealer external-action firewall changed: {key}")
    limitations = exact_keys(
        seal["limitations"],
        LEAN_SEAL_LIMITATION_KEYS,
        "lean post-generation seal limitations",
    )
    if (
        limitations["generation_target_freedom_basis"]
        != "FROZEN_RUNTIME_CONTROL_FLOW_PLUS_AUTHENTICATED_RUNTIME_RECEIPT_DECLARATION__NOT_EXTERNAL_PROCESS_OBSERVATION"
        or limitations["sentinel_parity_basis"]
        != "AUTHENTICATED_RUNTIME_RECEIPT_DECLARATION__NOT_SEALER_REEXECUTION"
        or limitations["all_row_dual_parity_claimed"] is not False
        or limitations["hostile_interpreter_security_claim"] is not False
        or limitations["review_independence_cryptographically_proven"] is not False
    ):
        raise AuthorityError("lean post-generation seal overclaims its evidence")
    return seal


def validate_lean_post_generation_seal_review_object(
    value: Any, *, seal_sha256: str, seal: dict[str, Any]
) -> dict[str, Any]:
    review = exact_keys(
        value, LEAN_POST_SEAL_REVIEW_KEYS, "lean post-generation seal review"
    )
    if (
        review["schema"]
        != "public-q24-signed-pair-lean-three-bank-post-generation-seal-independent-review-v4"
        or review["status"]
        != "GO_AUTHENTICATED_LEAN_RUNTIME_V4_TARGET_BLIND_THREE_BANKS"
        or review["post_generation_seal_sha256"] != seal_sha256
    ):
        raise AuthorityError("lean post-generation independent review mismatch")
    for key in (
        "pre_generation_source_authority_sha256",
        "pre_generation_source_authority_review_sha256",
        "pre_generation_v3_terminal_successor_record_sha256",
        "v3_terminal_failure_bindings",
        "manifest_compatibility_scan_receipt_sha256",
        "runtime_v4_static_bindings",
        "generation_transaction_bindings",
        "ordered_scope_bindings",
        "lean_generation_law",
        "bank_bindings",
    ):
        if review[key] != seal[key]:
            raise AuthorityError(f"lean post-generation review binding mismatch: {key}")
    canonical_id(review["review_id"], "lean post-generation review_id")
    canonical_id(
        review["issued_by"],
        "lean post-generation review issued_by",
        exact="independent_reviewer",
    )
    issued = canonical_utc(review["issued_utc"], "lean post-generation review issued_utc")
    if issued < seal["issued_utc"]:
        raise AuthorityError("lean post-generation review predates seal")
    for key in (
        "reviewed_after_generation",
        "receipt_and_all_six_bank_files_independently_rehashed",
        "sector_block_prefix_law_independently_replayed",
        "sentinel_scope_and_no_all_row_overclaim_verified",
        "manifest_compatibility_scan_authenticated",
        "v3_terminal_failure_and_no_retry_authenticated",
        "sealer_access_accounting_verified",
        "runtime_target_free_declaration_authenticated_not_observed",
        "sentinel_parity_authenticated_not_reexecuted",
        "generation_provenance_authenticated",
        "reviewer_different_from_seal_issuer",
    ):
        if review[key] is not True:
            raise AuthorityError(f"lean post-generation review boolean mismatch: {key}")
    if review["hostile_interpreter_security_claim"] is not False:
        raise AuthorityError("lean post-generation review overclaims hostile-interpreter security")
    return review


def validate_lean_v5_post_generation_seal_object(value: Any) -> dict[str, Any]:
    """Authenticate the V5 control overlay without changing V4 bank science."""
    seal = exact_keys(
        value, V5_LEAN_POST_SEAL_KEYS, "lean V5 post-generation seal"
    )
    if (
        seal["schema"]
        != "public-q24-signed-pair-lean-three-bank-post-generation-seal-v5"
        or seal["status"]
        != "SEALED_LEAN_RUNTIME_V5_TARGET_BLIND_THREE_BANK_GENERATION"
    ):
        raise AuthorityError("lean V5 post-generation seal schema/status mismatch")
    canonical_id(seal["seal_id"], "lean V5 post-generation seal_id")
    canonical_id(
        seal["issued_by"],
        "lean V5 post-generation issued_by",
        exact="post_generation_sealer",
    )
    canonical_utc(seal["issued_utc"], "lean V5 post-generation issued_utc")
    expected_direct = {
        "pre_generation_source_authority_sha256": V5_SOURCE_AUTHORITY_SHA256,
        "pre_generation_source_authority_review_sha256": (
            V5_SOURCE_AUTHORITY_REVIEW_SHA256
        ),
        "pre_generation_v4_terminal_successor_record_sha256": (
            V5_SUCCESSOR_RECORD_SHA256
        ),
        "independent_incident_root_cause_sha256": RUNTIME_V5_HASHES[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": RUNTIME_V5_HASHES[
            "flopscope_reset_verification_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": RUNTIME_V5_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ],
    }
    if any(
        not is_sha(seal[key]) or seal[key] != expected
        for key, expected in expected_direct.items()
    ):
        raise AuthorityError("lean V5 post-generation direct lineage changed")
    terminal = exact_keys(
        seal["v4_terminal_failure_bindings"],
        set(V4_TERMINAL_FAILURE_BINDINGS),
        "V4 terminal failure bindings",
    )
    if terminal != V4_TERMINAL_FAILURE_BINDINGS:
        raise AuthorityError("lean V5 post-generation V4 terminal record changed")
    runtime = exact_keys(
        seal["runtime_v5_static_bindings"],
        set(RUNTIME_V5_HASHES),
        "lean runtime-V5 static bindings",
    )
    if runtime != RUNTIME_V5_HASHES:
        raise AuthorityError("lean post-generation seal runtime-V5 roots changed")
    transaction_bindings = exact_keys(
        seal["generation_transaction_bindings"],
        GENERATION_TRANSACTION_BINDING_KEYS,
        "lean V5 generation transaction bindings",
    )
    if any(not is_sha(item) for item in transaction_bindings.values()):
        raise AuthorityError("lean V5 post-generation transaction binding invalid")
    if (
        transaction_bindings["generation_single_use_capability_sha256"]
        != transaction_bindings["consumed_capability_sha256"]
    ):
        raise AuthorityError("lean V5 generation capability digest changed during consumption")
    scope = exact_keys(
        seal["ordered_scope_bindings"],
        LEAN_ROW_SCOPE_BINDING_KEYS,
        "lean V5 seal ordered/sentinel scope bindings",
    )
    if scope != LEAN_ORDERED_SCOPE_BINDINGS:
        raise AuthorityError("lean V5 seal ordered/sentinel scope changed")
    if seal["lean_generation_law"] != expected_lean_v5_generation_law():
        raise AuthorityError("lean V5 seal generation/reset law changed")
    banks = exact_keys(
        seal["bank_bindings"], SEAL_BANK_BINDING_KEYS, "lean V5 seal bank bindings"
    )
    validate_sha_map(banks["complete_bank_sha256"], "lean V5 complete bank hashes")
    validate_sha_map(banks["A15_zero_sha256"], "lean V5 A15 hashes")
    transaction = exact_keys(
        seal["transaction"], TRANSACTION_KEYS, "lean V5 generation transaction"
    )
    expected_transaction = {
        "claim_unique": True,
        "capability_input_had_one_link": True,
        "capability_consumed_by_same_inode_hard_link_then_unlink": True,
        "capability_copy_created_with_O_EXCL": False,
        "capability_inode_preserved": True,
        "capability_original_absent": True,
        "one_use": True,
        "retry": False,
    }
    if any(
        transaction[key] is not expected
        for key, expected in expected_transaction.items()
    ):
        raise AuthorityError("lean V5 post-generation transaction assertion changed")
    evidence = exact_keys(
        seal["verification_evidence"],
        V5_LEAN_SEAL_VERIFICATION_KEYS,
        "lean V5 post-generation verification evidence",
    )
    for key in V5_LEAN_SEAL_VERIFICATION_KEYS - {"all_row_dual_parity_claimed"}:
        if evidence[key] is not True:
            raise AuthorityError(f"lean V5 verification evidence incomplete: {key}")
    if evidence["all_row_dual_parity_claimed"] is not False:
        raise AuthorityError("lean V5 post-generation seal overclaims all-row parity")
    access = exact_keys(
        seal["access_accounting"], SEAL_ACCESS_KEYS, "lean V5 sealer access accounting"
    )
    for key in (
        "sealer_target_artifacts_opened",
        "sealer_target_values_decoded",
        "sealer_observer_selection_artifacts_opened",
    ):
        if strict_int(access[key], minimum=0) != 0:
            raise AuthorityError(f"lean V5 sealer target firewall changed: {key}")
    for key in ("network", "package", "submission"):
        if access[key] is not False:
            raise AuthorityError(f"lean V5 sealer external-action firewall changed: {key}")
    limitations = exact_keys(
        seal["limitations"],
        V5_LEAN_SEAL_LIMITATION_KEYS,
        "lean V5 post-generation seal limitations",
    )
    if (
        limitations["generation_target_freedom_basis"]
        != "FROZEN_RUNTIME_CONTROL_FLOW_PLUS_AUTHENTICATED_RUNTIME_RECEIPT_DECLARATION__NOT_EXTERNAL_PROCESS_OBSERVATION"
        or limitations["sentinel_parity_basis"]
        != "AUTHENTICATED_RUNTIME_RECEIPT_DECLARATION__NOT_SEALER_REEXECUTION"
        or limitations["flopscope_reset_basis"]
        != "FROZEN_RUNTIME_V5_CONTROL_FLOW_PLUS_RESET_VERIFICATION_AND_REVIEW__NOT_EXTERNAL_PROCESS_OBSERVATION"
        or limitations["flopscope_reset_execution_independently_observed"] is not False
        or limitations["all_row_dual_parity_claimed"] is not False
        or limitations["hostile_interpreter_security_claim"] is not False
        or limitations["review_independence_cryptographically_proven"] is not False
    ):
        raise AuthorityError("lean V5 post-generation seal overclaims its evidence")
    return seal


def validate_lean_v5_post_generation_seal_review_object(
    value: Any, *, seal_sha256: str, seal: dict[str, Any]
) -> dict[str, Any]:
    review = exact_keys(
        value,
        V5_LEAN_POST_SEAL_REVIEW_KEYS,
        "lean V5 post-generation seal review",
    )
    if (
        review["schema"]
        != "public-q24-signed-pair-lean-three-bank-post-generation-seal-independent-review-v5"
        or review["status"]
        != "GO_AUTHENTICATED_LEAN_RUNTIME_V5_TARGET_BLIND_THREE_BANKS"
        or review["post_generation_seal_sha256"] != seal_sha256
    ):
        raise AuthorityError("lean V5 post-generation independent review mismatch")
    for key in (
        "pre_generation_source_authority_sha256",
        "pre_generation_source_authority_review_sha256",
        "pre_generation_v4_terminal_successor_record_sha256",
        "v4_terminal_failure_bindings",
        "independent_incident_root_cause_sha256",
        "flopscope_reset_verification_sha256",
        "manifest_compatibility_scan_receipt_sha256",
        "runtime_v5_static_bindings",
        "generation_transaction_bindings",
        "ordered_scope_bindings",
        "lean_generation_law",
        "bank_bindings",
    ):
        if review[key] != seal[key]:
            raise AuthorityError(
                f"lean V5 post-generation review binding mismatch: {key}"
            )
    canonical_id(review["review_id"], "lean V5 post-generation review_id")
    canonical_id(
        review["issued_by"],
        "lean V5 post-generation review issued_by",
        exact="independent_reviewer",
    )
    issued = canonical_utc(
        review["issued_utc"], "lean V5 post-generation review issued_utc"
    )
    if issued < seal["issued_utc"]:
        raise AuthorityError("lean V5 post-generation review predates seal")
    for key in (
        "reviewed_after_generation",
        "receipt_and_all_six_bank_files_independently_rehashed",
        "sector_block_prefix_law_independently_replayed",
        "sentinel_scope_and_no_all_row_overclaim_verified",
        "generation_receipt_and_bank_keysets_exact_v4_verified",
        "manifest_compatibility_scan_authenticated",
        "v4_terminal_failure_and_no_retry_authenticated",
        "independent_incident_root_cause_authenticated",
        "flopscope_post_commit_reset_law_authenticated_not_reexecuted",
        "sealer_access_accounting_verified",
        "runtime_target_free_declaration_authenticated_not_observed",
        "sentinel_parity_authenticated_not_reexecuted",
        "generation_provenance_authenticated",
        "reviewer_different_from_seal_issuer",
    ):
        if review[key] is not True:
            raise AuthorityError(f"lean V5 post-generation review boolean mismatch: {key}")
    if review["hostile_interpreter_security_claim"] is not False:
        raise AuthorityError(
            "lean V5 post-generation review overclaims hostile-interpreter security"
        )
    return review


RECEIPT_KEYS = {
    "schema",
    "status",
    "authority_chain",
    "callable_component_sha256",
    "claim_sha256",
    "runtime_token_receipt",
    "release_bindings",
    "anchor_bindings",
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
    "ordered_ranks",
    "row_receipts",
    "complete_banks",
    "resource_policy",
    "fourteen_parent_law_synchronizations_allowed_for_accuracy_science",
    "dual_stack_timing_is_deployment_only",
    "target_accessed",
    "training_run",
    "estimator_deployed",
    "package_built",
    "submission_made",
}
RECEIPT_AUTHORITY_CHAIN_KEYS = {
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "independent_review_sha256",
    "authority_anchor_sha256",
    "single_use_capability_sha256",
    "final_release_sha256",
    "acyclic",
    "capability_binds_final_release",
    "bundle_binds_transaction_authority",
}
RUNTIME_TOKEN_RECEIPT_KEYS = {
    "claim_sha256",
    "consumed_capability_sha256",
    "in_process_identity_validated",
    "serializable",
}
RUNTIME_BINDING_KEYS = {
    "runner_sha256",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "independent_review_sha256",
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
}
V4_RUNTIME_BINDING_KEYS = RUNTIME_BINDING_KEYS | {
    "manifest_compatibility_scan_receipt_sha256"
}
ROW_RECEIPT_KEYS = {
    "rank",
    "mlp_id",
    "row_manifest_sha256",
    "weights_sha256",
    "weights_schema_sha256",
    "capture_layers_zero_based",
    "same_layer_preactivation_alpha",
    "capture_after_full_reset_before_reset_stars",
    "A15_exact_zero_float32",
    "cross_rung_H16_prefix_byte_identical",
    "source_action_comparator",
    "five_component_parent_comparator",
    "banks",
    "exact_meters",
    "resource_observation",
}
ROW_BANK_KEYS = {
    "shape",
    "dtype",
    "finite",
    "action_bytes_sha256",
    "layer_action_bytes_sha256",
    "A15_zero_bytes_sha256",
}
ROW_METER_KEYS = {
    "complete_flops",
    "complete_calls",
    "base_parent_flops",
    "base_parent_calls",
    "capture_parent_flops",
    "capture_parent_calls",
    "literal_source_hook_flops",
    "literal_source_hook_calls",
    "reference_source_hook_flops",
    "reference_source_hook_calls",
    "literal_workspace_flops",
    "literal_workspace_calls",
    "reference_workspace_flops",
    "reference_workspace_calls",
    "peak_rss_bytes",
    "wall_seconds",
    "residual_seconds",
    "source_hook_capture_count",
    "meter_scope",
}
ROW_FLOAT_METER_KEYS = {"wall_seconds", "residual_seconds"}
ROW_INTEGER_METER_KEYS = ROW_METER_KEYS - ROW_FLOAT_METER_KEYS - {"meter_scope"}
FINAL_BANK_KEYS = {
    "row_count",
    "ranks",
    "banks",
    "exact_meters",
    "cross_rung_H16_prefix_byte_identical",
    "literal_reference_bytes_equal_all_rows",
    "base_capture_five_components_equal_all_rows",
    "resource_policy",
    "capability_consumption",
}
COMPLETE_BANK_ITEM_KEYS = {
    "path",
    "shape",
    "dtype",
    "finite",
    "sha256",
    "A15_zero_path",
    "A15_zero_sha256",
}
FINAL_METER_KEYS = {
    "complete_flops_by_rank",
    "complete_calls_by_rank",
    "base_parent_flops_by_rank",
    "base_parent_calls_by_rank",
    "capture_parent_flops_by_rank",
    "capture_parent_calls_by_rank",
    "literal_source_hook_flops_by_rank",
    "literal_source_hook_calls_by_rank",
    "reference_source_hook_flops_by_rank",
    "reference_source_hook_calls_by_rank",
    "literal_workspace_flops_by_rank",
    "literal_workspace_calls_by_rank",
    "reference_workspace_flops_by_rank",
    "reference_workspace_calls_by_rank",
    "peak_rss_bytes_by_rank",
    "wall_seconds_by_rank",
    "residual_seconds_by_rank",
}
FINAL_FLOAT_METER_KEYS = {
    "wall_seconds_by_rank",
    "residual_seconds_by_rank",
}
FINAL_INTEGER_METER_KEYS = FINAL_METER_KEYS - FINAL_FLOAT_METER_KEYS
SOURCE_ACTION_COMPARATOR_KEYS = {
    "comparator",
    "literal_implementation_id",
    "reference_implementation_id",
    "literal_source_sha256",
    "reference_source_sha256",
    "literal",
    "reference",
    "all_action_and_A15_bytes_equal",
}
PARENT_COMPARATOR_KEYS = {
    "comparator",
    "component_names",
    "base_component_sha256",
    "capture_component_sha256",
    "all_five_component_bytes_equal",
}
RESOURCE_OBSERVATION_KEYS = {
    "classification",
    "deployment_flop_cap",
    "complete_flops_within_deployment_cap",
    "peak_rss_bytes",
    "wall_seconds",
    "residual_seconds",
    "accuracy_veto_allowed",
}
FIVE_COMPONENT_ORDER = (
    "host",
    "native_raw20",
    "native_sigma",
    "native_j_correction",
    "final_public_prediction",
)
FIVE_COMPONENT_NAMES = set(FIVE_COMPONENT_ORDER)
SENTINEL_RANKS = (0, 343, 687)
LEAN_RECEIPT_KEYS = (
    RECEIPT_KEYS
    - {
        "dual_stack_timing_is_deployment_only",
    }
) | {
    "lean_execution_policy",
    "sentinel_duplicate_timing_is_deployment_only",
}
LEAN_ROW_RECEIPT_KEYS = ROW_RECEIPT_KEYS | {
    "sentinel_rank",
    "comparator_mode",
    "h256_prefix_derivation",
}
H256_DERIVATION_KEYS = {
    "method",
    "source_h",
    "derived_h",
    "all_derived_bytes_match_written_row",
    "sentinel_derived_matches_literal_and_reference",
}
LEAN_EXECUTION_POLICY_KEYS = {
    "captured_parent_and_literal_source_all_ranks",
    "unhooked_base_and_reference_source_sentinel_ranks",
    "literal_reference_parity_scope",
    "base_capture_parity_scope",
    "H16_H64_materialization",
    "all_688_rows_banked",
}
LEAN_FINAL_BANK_KEYS = (
    FINAL_BANK_KEYS
    - {
        "literal_reference_bytes_equal_all_rows",
        "base_capture_five_components_equal_all_rows",
    }
) | {
    "sentinel_ranks",
    "h256_prefix_derivation_complete_all_rows",
    "literal_reference_bytes_equal_sentinels",
    "base_capture_five_components_equal_sentinels",
}
LEAN_ROW_METER_KEYS = ROW_METER_KEYS | {"reference_source_hook_capture_count"}
LEAN_ROW_INTEGER_METER_KEYS = (
    LEAN_ROW_METER_KEYS - ROW_FLOAT_METER_KEYS - {"meter_scope"}
)
LEAN_FINAL_METER_KEYS = FINAL_METER_KEYS | {
    "reference_source_hook_capture_count_by_rank"
}
LEAN_FINAL_INTEGER_METER_KEYS = (
    LEAN_FINAL_METER_KEYS - FINAL_FLOAT_METER_KEYS
)


def _finite_nonnegative_number(value: Any, label: str) -> None:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise AuthorityError(f"{label} must be an exact JSON number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise AuthorityError(f"{label} must be finite and nonnegative")


def validate_runtime_v2_generation_receipt_object(
    value: Any,
) -> tuple[dict[str, Any], list[int]]:
    receipt = exact_keys(value, RECEIPT_KEYS, "three-bank generation receipt")
    if (
        receipt["schema"] != "public-q24-signed-pair-three-bank-generation-receipt-v2"
        or receipt["status"] != "SEALED_TARGET_BLIND_THREE_BANKS"
        or receipt["target_accessed"] is not False
        or receipt["training_run"] is not False
        or receipt["estimator_deployed"] is not False
        or receipt["package_built"] is not False
        or receipt["submission_made"] is not False
        or receipt["fourteen_parent_law_synchronizations_allowed_for_accuracy_science"]
        is not True
        or receipt["dual_stack_timing_is_deployment_only"] is not True
        or receipt["resource_policy"]
        != (
            "all FLOP/call/RSS/wall/residual meters are observational for "
            "accuracy science; any resource miss is deployment-only and cannot "
            "veto, erase, or alter a correct bank"
        )
    ):
        raise AuthorityError("exact runtime-V2 receipt schema/policy changed")
    chain = exact_keys(
        receipt["authority_chain"],
        RECEIPT_AUTHORITY_CHAIN_KEYS,
        "runtime-V2 receipt authority chain",
    )
    for key in RECEIPT_AUTHORITY_CHAIN_KEYS - {
        "acyclic",
        "capability_binds_final_release",
        "bundle_binds_transaction_authority",
    }:
        if not is_sha(chain[key]):
            raise AuthorityError(f"runtime-V2 authority-chain hash invalid: {key}")
    if (
        chain["callable_bundle_manifest_sha256"]
        != RUNTIME_V2_HASHES["callable_bundle_manifest_sha256"]
        or chain["execution_stack_manifest_sha256"]
        != RUNTIME_V2_HASHES["execution_stack_manifest_sha256"]
        or chain["independent_review_sha256"]
        != RUNTIME_V2_HASHES["independent_review_sha256"]
        or chain["acyclic"] is not True
        or chain["capability_binds_final_release"] is not False
        or chain["bundle_binds_transaction_authority"] is not False
    ):
        raise AuthorityError("runtime-V2 authority chain changed")
    components = exact_keys(
        receipt["callable_component_sha256"],
        set(RUNTIME_V2_COMPONENT_SHA256),
        "runtime-V2 callable components",
    )
    if components != RUNTIME_V2_COMPONENT_SHA256:
        raise AuthorityError("runtime-V2 callable component root changed")
    if not is_sha(receipt["claim_sha256"]):
        raise AuthorityError("runtime-V2 claim hash invalid")
    token = exact_keys(
        receipt["runtime_token_receipt"],
        RUNTIME_TOKEN_RECEIPT_KEYS,
        "runtime token receipt",
    )
    if (
        token["claim_sha256"] != receipt["claim_sha256"]
        or token["consumed_capability_sha256"]
        != chain["single_use_capability_sha256"]
        or token["in_process_identity_validated"] is not True
        or token["serializable"] is not False
    ):
        raise AuthorityError("runtime token receipt changed")
    expected_runtime_bindings = {
        "runner_sha256": RUNTIME_V2_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": RUNTIME_V2_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": RUNTIME_V2_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "independent_review_sha256": RUNTIME_V2_HASHES[
            "independent_review_sha256"
        ],
        "ordered_688_row_scope_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_row_scope_sha256"
        ],
        "ordered_688_record_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_record_sha256"
        ],
    }
    release_bindings = exact_keys(
        receipt["release_bindings"], RUNTIME_BINDING_KEYS, "release bindings"
    )
    anchor_bindings = exact_keys(
        receipt["anchor_bindings"], RUNTIME_BINDING_KEYS, "anchor bindings"
    )
    if release_bindings != expected_runtime_bindings or anchor_bindings != expected_runtime_bindings:
        raise AuthorityError("runtime-V2 receipt release/anchor bindings changed")
    if (
        receipt["ordered_688_row_scope_sha256"]
        != ORDERED_SCOPE_BINDINGS["ordered_688_row_scope_sha256"]
        or receipt["ordered_688_record_sha256"]
        != ORDERED_SCOPE_BINDINGS["ordered_688_record_sha256"]
    ):
        raise AuthorityError("three-bank receipt ordered scope changed")
    validate_ordered_ranks(receipt["ordered_ranks"])

    rows = receipt["row_receipts"]
    if type(rows) is not list or len(rows) != ROWS:
        raise AuthorityError("generation receipt must contain exactly 688 rows")
    mlp_ids: list[int] = []
    for expected_rank, candidate in enumerate(rows):
        row = exact_keys(candidate, ROW_RECEIPT_KEYS, "row receipt")
        rank = strict_int(row["rank"], minimum=0, maximum=ROWS - 1)
        mlp_id = strict_int(row["mlp_id"], minimum=0)
        if rank != expected_rank:
            raise AuthorityError("row receipt rank order changed")
        mlp_ids.append(mlp_id)
        for key in ("row_manifest_sha256", "weights_sha256", "weights_schema_sha256"):
            if not is_sha(row[key]):
                raise AuthorityError(f"row receipt identity hash invalid: {key}")
        if row["capture_layers_zero_based"] != list(ACTION_LAYERS):
            raise AuthorityError("row receipt capture layers changed")
        for key in (
            "same_layer_preactivation_alpha",
            "capture_after_full_reset_before_reset_stars",
            "A15_exact_zero_float32",
            "cross_rung_H16_prefix_byte_identical",
        ):
            if row[key] is not True:
                raise AuthorityError(f"row receipt exact boolean changed: {key}")
        comparator = exact_keys(
            row["source_action_comparator"],
            SOURCE_ACTION_COMPARATOR_KEYS,
            "source-action comparator",
        )
        if (
            comparator["comparator"]
            != "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE"
            or comparator["literal_implementation_id"]
            != "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2"
            or comparator["reference_implementation_id"]
            != "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2"
            or comparator["literal_source_sha256"]
            != RUNTIME_V2_COMPONENT_SHA256["signed_pair_literal.py"]
            or comparator["reference_source_sha256"]
            != RUNTIME_V2_COMPONENT_SHA256["signed_pair_reference.py"]
            or comparator["all_action_and_A15_bytes_equal"] is not True
        ):
            raise AuthorityError("source-action comparator changed")
        literal = exact_keys(comparator["literal"], {"16", "64", "256"}, "literal comparator")
        reference = exact_keys(comparator["reference"], {"16", "64", "256"}, "reference comparator")
        parent = exact_keys(
            row["five_component_parent_comparator"],
            PARENT_COMPARATOR_KEYS,
            "five-component parent comparator",
        )
        if (
            parent["comparator"] != "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE"
            or parent["component_names"] != list(FIVE_COMPONENT_ORDER)
            or parent["all_five_component_bytes_equal"] is not True
        ):
            raise AuthorityError("five-component parent comparator changed")
        base_components = exact_keys(
            parent["base_component_sha256"], FIVE_COMPONENT_NAMES, "base component hashes"
        )
        capture_components = exact_keys(
            parent["capture_component_sha256"], FIVE_COMPONENT_NAMES, "capture component hashes"
        )
        if base_components != capture_components or any(
            not is_sha(item) for item in base_components.values()
        ):
            raise AuthorityError("five-component parent comparator failed")
        banks = exact_keys(row["banks"], {"16", "64", "256"}, "row banks")
        for h in HS:
            item = exact_keys(banks[str(h)], ROW_BANK_KEYS, "row bank")
            if (
                item["shape"] != [13, WIDTH, h]
                or item["dtype"] != "float32"
                or item["finite"] is not True
                or not is_sha(item["action_bytes_sha256"])
                or not is_sha(item["A15_zero_bytes_sha256"])
            ):
                raise AuthorityError("row bank schema changed")
            layers = item["layer_action_bytes_sha256"]
            if (
                type(layers) is not list
                or len(layers) != 13
                or any(not is_sha(item_hash) for item_hash in layers)
            ):
                raise AuthorityError("row bank layer hashes changed")
            for side in (literal, reference):
                comparison = exact_keys(
                    side[str(h)],
                    {"action_sha256", "A15_zero_sha256"},
                    "source comparator H item",
                )
                if (
                    comparison["action_sha256"] != item["action_bytes_sha256"]
                    or comparison["A15_zero_sha256"]
                    != item["A15_zero_bytes_sha256"]
                ):
                    raise AuthorityError("source comparator/output bank mismatch")
            if literal[str(h)] != reference[str(h)]:
                raise AuthorityError("literal/reference row bytes differ")
        meters = exact_keys(row["exact_meters"], ROW_METER_KEYS, "row meters")
        for key in ROW_INTEGER_METER_KEYS:
            strict_int(meters[key], minimum=0)
        for key in ROW_FLOAT_METER_KEYS:
            if type(meters[key]) is not float:
                raise AuthorityError(f"row meter {key} must be a JSON float")
            _finite_nonnegative_number(meters[key], f"row meter {key}")
        if (
            strict_int(meters["source_hook_capture_count"]) != 13
            or meters["meter_scope"]
            != "BASE_PARENT_PLUS_CAPTURE_PARENT_PLUS_LITERAL_AND_REFERENCE"
        ):
            raise AuthorityError("row meter scope changed")
        observation = exact_keys(
            row["resource_observation"],
            RESOURCE_OBSERVATION_KEYS,
            "row resource observation",
        )
        if (
            observation["classification"] != "OBSERVATIONAL_DEPLOYMENT_ONLY"
            or observation["deployment_flop_cap"] != 2_199_023_255_552
            or type(observation["complete_flops_within_deployment_cap"]) is not bool
            or observation["peak_rss_bytes"] != meters["peak_rss_bytes"]
            or observation["wall_seconds"] != meters["wall_seconds"]
            or observation["residual_seconds"] != meters["residual_seconds"]
            or observation["accuracy_veto_allowed"] is not False
        ):
            raise AuthorityError("row resource observation changed")
    if len(set(mlp_ids)) != ROWS:
        raise AuthorityError("generation receipt mlp_id roster is not unique")

    final = exact_keys(receipt["complete_banks"], FINAL_BANK_KEYS, "complete banks")
    if (
        strict_int(final["row_count"]) != ROWS
        or final["cross_rung_H16_prefix_byte_identical"] is not True
        or final["literal_reference_bytes_equal_all_rows"] is not True
        or final["base_capture_five_components_equal_all_rows"] is not True
        or final["resource_policy"]
        != "RESOURCE_AND_RESIDUAL_OBSERVATIONAL_DEPLOYMENT_ONLY"
    ):
        raise AuthorityError("complete bank declarations changed")
    validate_ordered_ranks(final["ranks"])
    complete = exact_keys(final["banks"], {"16", "64", "256"}, "complete bank map")
    for h in HS:
        item = exact_keys(
            complete[str(h)], COMPLETE_BANK_ITEM_KEYS, "complete bank item"
        )
        canonical_relative_path(item["path"], f"H{h} bank path")
        canonical_relative_path(item["A15_zero_path"], f"H{h} A15 path")
        if (
            item["shape"] != [ROWS, 13, WIDTH, h]
            or item["dtype"] != "float32"
            or item["finite"] is not True
            or not is_sha(item["sha256"])
            or not is_sha(item["A15_zero_sha256"])
        ):
            raise AuthorityError("complete bank item schema changed")
    meter_vectors = exact_keys(final["exact_meters"], FINAL_METER_KEYS, "meter vectors")
    for name, vector in meter_vectors.items():
        if type(vector) is not list or len(vector) != ROWS:
            raise AuthorityError(f"meter vector length changed: {name}")
        for rank, meter in enumerate(vector):
            scalar_name = name.removesuffix("_by_rank")
            if name in FINAL_INTEGER_METER_KEYS:
                strict_int(meter, minimum=0)
            else:
                if type(meter) is not float:
                    raise AuthorityError(f"meter vector {name} must contain JSON floats")
                _finite_nonnegative_number(meter, f"meter vector {name}")
            if meter != rows[rank]["exact_meters"][scalar_name]:
                raise AuthorityError(f"meter vector/row scalar mismatch: {name}")
    consumption = exact_keys(
        final["capability_consumption"],
        {
            "capability_id",
            "issued_by",
            "issued_utc",
            "consumed_path",
            "consumed_sha256",
            "original_absent",
            "one_use_inode",
            "one_use",
            "no_retry",
        },
        "runtime-V2 capability consumption",
    )
    canonical_id(consumption["capability_id"], "generation capability_id")
    canonical_id(consumption["issued_by"], "generation capability issuer")
    canonical_utc(consumption["issued_utc"], "generation capability issued_utc")
    if (
        not is_sha(consumption["consumed_sha256"])
        or consumption["consumed_sha256"] != chain["single_use_capability_sha256"]
        or consumption["original_absent"] is not True
        or consumption["one_use"] is not True
        or consumption["no_retry"] is not True
    ):
        raise AuthorityError("runtime-V2 capability consumption changed")
    inode = exact_keys(
        consumption["one_use_inode"],
        {"device", "inode", "size", "initial_link_count", "consumed_link_count"},
        "runtime-V2 one-use inode",
    )
    for name in ("device", "inode", "size"):
        strict_int(inode[name], minimum=0)
    if inode["initial_link_count"] != 1 or inode["consumed_link_count"] != 1:
        raise AuthorityError("runtime-V2 same-inode consumption link ledger changed")
    return receipt, mlp_ids


def validate_lean_runtime_v3_generation_receipt_object(
    value: Any,
) -> tuple[dict[str, Any], list[int]]:
    """Validate only the frozen all-row-bank/sentinel-parity runtime V3 receipt."""
    if set(RUNTIME_V3_HASHES) != set(RUNTIME_V2_HASHES) or set(
        RUNTIME_V3_COMPONENT_SHA256
    ) != set(RUNTIME_V2_COMPONENT_SHA256):
        raise AuthorityError("lean runtime-V3 roots are not frozen and complete")
    receipt = exact_keys(value, LEAN_RECEIPT_KEYS, "lean runtime-V3 receipt")
    if (
        receipt["schema"]
        != "public-q24-signed-pair-three-bank-generation-receipt-v3"
        or receipt["status"] != "SEALED_TARGET_BLIND_THREE_BANKS"
        or receipt["target_accessed"] is not False
        or receipt["training_run"] is not False
        or receipt["estimator_deployed"] is not False
        or receipt["package_built"] is not False
        or receipt["submission_made"] is not False
        or receipt[
            "fourteen_parent_law_synchronizations_allowed_for_accuracy_science"
        ]
        is not True
        or receipt["sentinel_duplicate_timing_is_deployment_only"] is not True
        or receipt["resource_policy"]
        != (
            "all FLOP/call/RSS/wall/residual meters are observational for "
            "accuracy science; any resource miss is deployment-only and cannot "
            "veto, erase, or alter a correct bank"
        )
    ):
        raise AuthorityError("exact lean runtime-V3 receipt schema/policy changed")

    policy = exact_keys(
        receipt["lean_execution_policy"],
        LEAN_EXECUTION_POLICY_KEYS,
        "lean execution policy",
    )
    if (
        policy["captured_parent_and_literal_source_all_ranks"] is not True
        or policy["unhooked_base_and_reference_source_sentinel_ranks"]
        != list(SENTINEL_RANKS)
        or policy["literal_reference_parity_scope"] != "SENTINELS_ONLY"
        or policy["base_capture_parity_scope"] != "SENTINELS_ONLY"
        or policy["H16_H64_materialization"]
        != "EXACT_EIGHT_SECTOR_RECEIVER_PREFIXES_FROM_H256_ALL_ROWS"
        or policy["all_688_rows_banked"] is not True
    ):
        raise AuthorityError("lean execution policy or sentinel scope changed")

    chain = exact_keys(
        receipt["authority_chain"],
        RECEIPT_AUTHORITY_CHAIN_KEYS,
        "lean runtime-V3 receipt authority chain",
    )
    for key in RECEIPT_AUTHORITY_CHAIN_KEYS - {
        "acyclic",
        "capability_binds_final_release",
        "bundle_binds_transaction_authority",
    }:
        if not is_sha(chain[key]):
            raise AuthorityError(f"lean runtime-V3 authority-chain hash invalid: {key}")
    if (
        chain["callable_bundle_manifest_sha256"]
        != RUNTIME_V3_HASHES["callable_bundle_manifest_sha256"]
        or chain["execution_stack_manifest_sha256"]
        != RUNTIME_V3_HASHES["execution_stack_manifest_sha256"]
        or chain["independent_review_sha256"]
        != RUNTIME_V3_HASHES["independent_review_sha256"]
        or chain["acyclic"] is not True
        or chain["capability_binds_final_release"] is not False
        or chain["bundle_binds_transaction_authority"] is not False
    ):
        raise AuthorityError("lean runtime-V3 authority chain changed")
    components = exact_keys(
        receipt["callable_component_sha256"],
        set(RUNTIME_V3_COMPONENT_SHA256),
        "lean runtime-V3 callable components",
    )
    if components != RUNTIME_V3_COMPONENT_SHA256:
        raise AuthorityError("lean runtime-V3 callable component root changed")
    if not is_sha(receipt["claim_sha256"]):
        raise AuthorityError("lean runtime-V3 claim hash invalid")
    token = exact_keys(
        receipt["runtime_token_receipt"],
        RUNTIME_TOKEN_RECEIPT_KEYS,
        "lean runtime-V3 token receipt",
    )
    if (
        token["claim_sha256"] != receipt["claim_sha256"]
        or token["consumed_capability_sha256"]
        != chain["single_use_capability_sha256"]
        or token["in_process_identity_validated"] is not True
        or token["serializable"] is not False
    ):
        raise AuthorityError("lean runtime-V3 token receipt changed")
    expected_runtime_bindings = {
        "runner_sha256": RUNTIME_V3_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": RUNTIME_V3_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": RUNTIME_V3_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "independent_review_sha256": RUNTIME_V3_HASHES[
            "independent_review_sha256"
        ],
        "ordered_688_row_scope_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_row_scope_sha256"
        ],
        "ordered_688_record_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_record_sha256"
        ],
    }
    release_bindings = exact_keys(
        receipt["release_bindings"], RUNTIME_BINDING_KEYS, "lean release bindings"
    )
    anchor_bindings = exact_keys(
        receipt["anchor_bindings"], RUNTIME_BINDING_KEYS, "lean anchor bindings"
    )
    if (
        release_bindings != expected_runtime_bindings
        or anchor_bindings != expected_runtime_bindings
    ):
        raise AuthorityError("lean runtime-V3 receipt release/anchor bindings changed")
    if (
        receipt["ordered_688_row_scope_sha256"]
        != ORDERED_SCOPE_BINDINGS["ordered_688_row_scope_sha256"]
        or receipt["ordered_688_record_sha256"]
        != ORDERED_SCOPE_BINDINGS["ordered_688_record_sha256"]
    ):
        raise AuthorityError("lean runtime-V3 ordered scope changed")
    validate_ordered_ranks(receipt["ordered_ranks"])

    rows = receipt["row_receipts"]
    if type(rows) is not list or len(rows) != ROWS:
        raise AuthorityError("lean runtime-V3 receipt must contain exactly 688 rows")
    mlp_ids: list[int] = []
    for expected_rank, candidate in enumerate(rows):
        row = exact_keys(candidate, LEAN_ROW_RECEIPT_KEYS, "lean row receipt")
        rank = strict_int(row["rank"], minimum=0, maximum=ROWS - 1)
        mlp_id = strict_int(row["mlp_id"], minimum=0)
        if rank != expected_rank:
            raise AuthorityError("lean row receipt rank order changed")
        sentinel = rank in SENTINEL_RANKS
        if row["sentinel_rank"] is not sentinel or row["comparator_mode"] != (
            "SENTINEL_DUAL_PARITY" if sentinel else "LEAN_CAPTURE_LITERAL_ONLY"
        ):
            raise AuthorityError("lean row sentinel/comparator mode changed")
        mlp_ids.append(mlp_id)
        for key in ("row_manifest_sha256", "weights_sha256", "weights_schema_sha256"):
            if not is_sha(row[key]):
                raise AuthorityError(f"lean row identity hash invalid: {key}")
        if row["capture_layers_zero_based"] != list(ACTION_LAYERS):
            raise AuthorityError("lean row capture layers changed")
        for key in (
            "same_layer_preactivation_alpha",
            "capture_after_full_reset_before_reset_stars",
            "A15_exact_zero_float32",
            "cross_rung_H16_prefix_byte_identical",
        ):
            if row[key] is not True:
                raise AuthorityError(f"lean row exact boolean changed: {key}")
        derivation = exact_keys(
            row["h256_prefix_derivation"],
            H256_DERIVATION_KEYS,
            "lean H256 prefix derivation",
        )
        if (
            derivation["method"] != "EIGHT_SECTOR_RECEIVER_PREFIX_FROM_H256"
            or derivation["source_h"] != 256
            or derivation["derived_h"] != [16, 64]
            or derivation["all_derived_bytes_match_written_row"] is not True
            or derivation["sentinel_derived_matches_literal_and_reference"]
            is not (True if sentinel else None)
        ):
            raise AuthorityError("lean H256 prefix derivation evidence changed")

        comparator = row["source_action_comparator"]
        parent = row["five_component_parent_comparator"]
        literal: dict[str, Any] | None = None
        reference: dict[str, Any] | None = None
        if sentinel:
            comparator = exact_keys(
                comparator,
                SOURCE_ACTION_COMPARATOR_KEYS,
                "lean sentinel source-action comparator",
            )
            if (
                comparator["comparator"]
                != "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE"
                or comparator["literal_implementation_id"]
                != "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2"
                or comparator["reference_implementation_id"]
                != "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2"
                or comparator["literal_source_sha256"]
                != RUNTIME_V3_COMPONENT_SHA256["signed_pair_literal.py"]
                or comparator["reference_source_sha256"]
                != RUNTIME_V3_COMPONENT_SHA256["signed_pair_reference.py"]
                or comparator["all_action_and_A15_bytes_equal"] is not True
            ):
                raise AuthorityError("lean sentinel source-action comparator changed")
            literal = exact_keys(
                comparator["literal"],
                {"16", "64", "256"},
                "lean sentinel literal comparator",
            )
            reference = exact_keys(
                comparator["reference"],
                {"16", "64", "256"},
                "lean sentinel reference comparator",
            )
            parent = exact_keys(
                parent,
                PARENT_COMPARATOR_KEYS,
                "lean sentinel parent comparator",
            )
            if (
                parent["comparator"]
                != "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE"
                or parent["component_names"] != list(FIVE_COMPONENT_ORDER)
                or parent["all_five_component_bytes_equal"] is not True
            ):
                raise AuthorityError("lean sentinel parent comparator changed")
            base_components = exact_keys(
                parent["base_component_sha256"],
                FIVE_COMPONENT_NAMES,
                "lean sentinel base component hashes",
            )
            capture_components = exact_keys(
                parent["capture_component_sha256"],
                FIVE_COMPONENT_NAMES,
                "lean sentinel capture component hashes",
            )
            if base_components != capture_components or any(
                not is_sha(item) for item in base_components.values()
            ):
                raise AuthorityError("lean sentinel parent comparator failed")
        elif comparator is not None or parent is not None:
            raise AuthorityError("lean non-sentinel row contains dual-parity evidence")

        banks = exact_keys(row["banks"], {"16", "64", "256"}, "lean row banks")
        for h in HS:
            item = exact_keys(banks[str(h)], ROW_BANK_KEYS, "lean row bank")
            if (
                item["shape"] != [13, WIDTH, h]
                or item["dtype"] != "float32"
                or item["finite"] is not True
                or not is_sha(item["action_bytes_sha256"])
                or not is_sha(item["A15_zero_bytes_sha256"])
            ):
                raise AuthorityError("lean row bank schema changed")
            layers = item["layer_action_bytes_sha256"]
            if (
                type(layers) is not list
                or len(layers) != 13
                or any(not is_sha(item_hash) for item_hash in layers)
            ):
                raise AuthorityError("lean row bank layer hashes changed")
            if sentinel:
                if literal is None or reference is None:
                    raise AuthorityError("lean sentinel comparator evidence absent")
                for side in (literal, reference):
                    comparison = exact_keys(
                        side[str(h)],
                        {"action_sha256", "A15_zero_sha256"},
                        "lean sentinel source comparator H item",
                    )
                    if (
                        comparison["action_sha256"]
                        != item["action_bytes_sha256"]
                        or comparison["A15_zero_sha256"]
                        != item["A15_zero_bytes_sha256"]
                    ):
                        raise AuthorityError(
                            "lean sentinel source comparator/output mismatch"
                        )
                if literal[str(h)] != reference[str(h)]:
                    raise AuthorityError("lean sentinel literal/reference bytes differ")

        meters = exact_keys(row["exact_meters"], LEAN_ROW_METER_KEYS, "lean row meters")
        for key in LEAN_ROW_INTEGER_METER_KEYS:
            strict_int(meters[key], minimum=0)
        for key in ROW_FLOAT_METER_KEYS:
            if type(meters[key]) is not float:
                raise AuthorityError(f"lean row meter {key} must be a JSON float")
            _finite_nonnegative_number(meters[key], f"lean row meter {key}")
        if (
            meters["source_hook_capture_count"] != 13
            or meters["reference_source_hook_capture_count"]
            != (13 if sentinel else 0)
            or meters["meter_scope"]
            != "CAPTURE_LITERAL_ALL_ROWS_BASE_REFERENCE_SENTINELS"
        ):
            raise AuthorityError("lean row meter scope/sentinel ledger changed")
        for key in (
            "base_parent_flops",
            "base_parent_calls",
            "reference_source_hook_flops",
            "reference_source_hook_calls",
            "reference_workspace_flops",
            "reference_workspace_calls",
        ):
            if not sentinel and meters[key] != 0:
                raise AuthorityError(f"lean non-sentinel duplicate meter nonzero: {key}")
        observation = exact_keys(
            row["resource_observation"],
            RESOURCE_OBSERVATION_KEYS,
            "lean row resource observation",
        )
        if (
            observation["classification"] != "OBSERVATIONAL_DEPLOYMENT_ONLY"
            or observation["deployment_flop_cap"] != 2_199_023_255_552
            or type(observation["complete_flops_within_deployment_cap"]) is not bool
            or observation["peak_rss_bytes"] != meters["peak_rss_bytes"]
            or observation["wall_seconds"] != meters["wall_seconds"]
            or observation["residual_seconds"] != meters["residual_seconds"]
            or observation["accuracy_veto_allowed"] is not False
        ):
            raise AuthorityError("lean row resource observation changed")
    if len(set(mlp_ids)) != ROWS:
        raise AuthorityError("lean runtime-V3 mlp_id roster is not unique")

    final = exact_keys(receipt["complete_banks"], LEAN_FINAL_BANK_KEYS, "lean complete banks")
    if (
        strict_int(final["row_count"]) != ROWS
        or final["sentinel_ranks"] != list(SENTINEL_RANKS)
        or final["cross_rung_H16_prefix_byte_identical"] is not True
        or final["h256_prefix_derivation_complete_all_rows"] is not True
        or final["literal_reference_bytes_equal_sentinels"] is not True
        or final["base_capture_five_components_equal_sentinels"] is not True
        or final["resource_policy"]
        != "RESOURCE_AND_RESIDUAL_OBSERVATIONAL_DEPLOYMENT_ONLY"
    ):
        raise AuthorityError("lean complete-bank declarations changed")
    validate_ordered_ranks(final["ranks"])
    complete = exact_keys(final["banks"], {"16", "64", "256"}, "lean complete bank map")
    for h in HS:
        item = exact_keys(
            complete[str(h)], COMPLETE_BANK_ITEM_KEYS, "lean complete bank item"
        )
        canonical_relative_path(item["path"], f"lean H{h} bank path")
        canonical_relative_path(item["A15_zero_path"], f"lean H{h} A15 path")
        if (
            item["shape"] != [ROWS, 13, WIDTH, h]
            or item["dtype"] != "float32"
            or item["finite"] is not True
            or not is_sha(item["sha256"])
            or not is_sha(item["A15_zero_sha256"])
        ):
            raise AuthorityError("lean complete bank item schema changed")
    meter_vectors = exact_keys(
        final["exact_meters"], LEAN_FINAL_METER_KEYS, "lean meter vectors"
    )
    for name, vector in meter_vectors.items():
        if type(vector) is not list or len(vector) != ROWS:
            raise AuthorityError(f"lean meter vector length changed: {name}")
        for rank, meter in enumerate(vector):
            scalar_name = name.removesuffix("_by_rank")
            if name in LEAN_FINAL_INTEGER_METER_KEYS:
                strict_int(meter, minimum=0)
            else:
                if type(meter) is not float:
                    raise AuthorityError(
                        f"lean meter vector {name} must contain JSON floats"
                    )
                _finite_nonnegative_number(meter, f"lean meter vector {name}")
            if meter != rows[rank]["exact_meters"][scalar_name]:
                raise AuthorityError(f"lean meter vector/row scalar mismatch: {name}")
    consumption = exact_keys(
        final["capability_consumption"],
        {
            "capability_id",
            "issued_by",
            "issued_utc",
            "consumed_path",
            "consumed_sha256",
            "original_absent",
            "one_use_inode",
            "one_use",
            "no_retry",
        },
        "lean runtime-V3 capability consumption",
    )
    canonical_id(consumption["capability_id"], "lean generation capability_id")
    canonical_id(consumption["issued_by"], "lean generation capability issuer")
    canonical_utc(consumption["issued_utc"], "lean generation capability issued_utc")
    if (
        not is_sha(consumption["consumed_sha256"])
        or consumption["consumed_sha256"]
        != chain["single_use_capability_sha256"]
        or consumption["original_absent"] is not True
        or consumption["one_use"] is not True
        or consumption["no_retry"] is not True
    ):
        raise AuthorityError("lean runtime-V3 capability consumption changed")
    inode = exact_keys(
        consumption["one_use_inode"],
        {"device", "inode", "size", "initial_link_count", "consumed_link_count"},
        "lean runtime-V3 one-use inode",
    )
    for name in ("device", "inode", "size"):
        strict_int(inode[name], minimum=0)
    if inode["initial_link_count"] != 1 or inode["consumed_link_count"] != 1:
        raise AuthorityError("lean runtime-V3 same-inode consumption ledger changed")
    return receipt, mlp_ids


def validate_lean_runtime_v4_generation_receipt_object(
    value: Any,
) -> tuple[dict[str, Any], list[int]]:
    """Validate only the frozen all-row-bank/sentinel-parity runtime V4 receipt."""
    if set(RUNTIME_V4_COMPONENT_SHA256) != set(RUNTIME_V2_COMPONENT_SHA256):
        raise AuthorityError("lean runtime-V4 roots are not frozen and complete")
    receipt = exact_keys(value, LEAN_RECEIPT_KEYS, "lean runtime-V4 receipt")
    if (
        receipt["schema"]
        != "public-q24-signed-pair-three-bank-generation-receipt-v4"
        or receipt["status"] != "SEALED_TARGET_BLIND_THREE_BANKS"
        or receipt["target_accessed"] is not False
        or receipt["training_run"] is not False
        or receipt["estimator_deployed"] is not False
        or receipt["package_built"] is not False
        or receipt["submission_made"] is not False
        or receipt[
            "fourteen_parent_law_synchronizations_allowed_for_accuracy_science"
        ]
        is not True
        or receipt["sentinel_duplicate_timing_is_deployment_only"] is not True
        or receipt["resource_policy"]
        != (
            "all FLOP/call/RSS/wall/residual meters are observational for "
            "accuracy science; any resource miss is deployment-only and cannot "
            "veto, erase, or alter a correct bank"
        )
    ):
        raise AuthorityError("exact lean runtime-V4 receipt schema/policy changed")

    policy = exact_keys(
        receipt["lean_execution_policy"],
        LEAN_EXECUTION_POLICY_KEYS,
        "lean execution policy",
    )
    if (
        policy["captured_parent_and_literal_source_all_ranks"] is not True
        or policy["unhooked_base_and_reference_source_sentinel_ranks"]
        != list(SENTINEL_RANKS)
        or policy["literal_reference_parity_scope"] != "SENTINELS_ONLY"
        or policy["base_capture_parity_scope"] != "SENTINELS_ONLY"
        or policy["H16_H64_materialization"]
        != "EXACT_EIGHT_SECTOR_RECEIVER_PREFIXES_FROM_H256_ALL_ROWS"
        or policy["all_688_rows_banked"] is not True
    ):
        raise AuthorityError("lean execution policy or sentinel scope changed")

    chain = exact_keys(
        receipt["authority_chain"],
        RECEIPT_AUTHORITY_CHAIN_KEYS,
        "lean runtime-V4 receipt authority chain",
    )
    for key in RECEIPT_AUTHORITY_CHAIN_KEYS - {
        "acyclic",
        "capability_binds_final_release",
        "bundle_binds_transaction_authority",
    }:
        if not is_sha(chain[key]):
            raise AuthorityError(f"lean runtime-V4 authority-chain hash invalid: {key}")
    if (
        chain["callable_bundle_manifest_sha256"]
        != RUNTIME_V4_HASHES["callable_bundle_manifest_sha256"]
        or chain["execution_stack_manifest_sha256"]
        != RUNTIME_V4_HASHES["execution_stack_manifest_sha256"]
        or chain["independent_review_sha256"]
        != RUNTIME_V4_HASHES["independent_review_sha256"]
        or chain["acyclic"] is not True
        or chain["capability_binds_final_release"] is not False
        or chain["bundle_binds_transaction_authority"] is not False
    ):
        raise AuthorityError("lean runtime-V4 authority chain changed")
    components = exact_keys(
        receipt["callable_component_sha256"],
        set(RUNTIME_V4_COMPONENT_SHA256),
        "lean runtime-V4 callable components",
    )
    if components != RUNTIME_V4_COMPONENT_SHA256:
        raise AuthorityError("lean runtime-V4 callable component root changed")
    if not is_sha(receipt["claim_sha256"]):
        raise AuthorityError("lean runtime-V4 claim hash invalid")
    token = exact_keys(
        receipt["runtime_token_receipt"],
        RUNTIME_TOKEN_RECEIPT_KEYS,
        "lean runtime-V4 token receipt",
    )
    if (
        token["claim_sha256"] != receipt["claim_sha256"]
        or token["consumed_capability_sha256"]
        != chain["single_use_capability_sha256"]
        or token["in_process_identity_validated"] is not True
        or token["serializable"] is not False
    ):
        raise AuthorityError("lean runtime-V4 token receipt changed")
    expected_runtime_bindings = {
        "runner_sha256": RUNTIME_V4_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": RUNTIME_V4_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": RUNTIME_V4_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "independent_review_sha256": RUNTIME_V4_HASHES[
            "independent_review_sha256"
        ],
        "ordered_688_row_scope_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_row_scope_sha256"
        ],
        "ordered_688_record_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_record_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": RUNTIME_V4_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ],
    }
    release_bindings = exact_keys(
        receipt["release_bindings"], V4_RUNTIME_BINDING_KEYS, "lean release bindings"
    )
    anchor_bindings = exact_keys(
        receipt["anchor_bindings"], V4_RUNTIME_BINDING_KEYS, "lean anchor bindings"
    )
    if (
        release_bindings != expected_runtime_bindings
        or anchor_bindings != expected_runtime_bindings
    ):
        raise AuthorityError("lean runtime-V4 receipt release/anchor bindings changed")
    if (
        receipt["ordered_688_row_scope_sha256"]
        != ORDERED_SCOPE_BINDINGS["ordered_688_row_scope_sha256"]
        or receipt["ordered_688_record_sha256"]
        != ORDERED_SCOPE_BINDINGS["ordered_688_record_sha256"]
    ):
        raise AuthorityError("lean runtime-V4 ordered scope changed")
    validate_ordered_ranks(receipt["ordered_ranks"])

    rows = receipt["row_receipts"]
    if type(rows) is not list or len(rows) != ROWS:
        raise AuthorityError("lean runtime-V4 receipt must contain exactly 688 rows")
    mlp_ids: list[int] = []
    for expected_rank, candidate in enumerate(rows):
        row = exact_keys(candidate, LEAN_ROW_RECEIPT_KEYS, "lean row receipt")
        rank = strict_int(row["rank"], minimum=0, maximum=ROWS - 1)
        mlp_id = strict_int(row["mlp_id"], minimum=0)
        if rank != expected_rank:
            raise AuthorityError("lean row receipt rank order changed")
        sentinel = rank in SENTINEL_RANKS
        if row["sentinel_rank"] is not sentinel or row["comparator_mode"] != (
            "SENTINEL_DUAL_PARITY" if sentinel else "LEAN_CAPTURE_LITERAL_ONLY"
        ):
            raise AuthorityError("lean row sentinel/comparator mode changed")
        mlp_ids.append(mlp_id)
        for key in ("row_manifest_sha256", "weights_sha256", "weights_schema_sha256"):
            if not is_sha(row[key]):
                raise AuthorityError(f"lean row identity hash invalid: {key}")
        if row["capture_layers_zero_based"] != list(ACTION_LAYERS):
            raise AuthorityError("lean row capture layers changed")
        for key in (
            "same_layer_preactivation_alpha",
            "capture_after_full_reset_before_reset_stars",
            "A15_exact_zero_float32",
            "cross_rung_H16_prefix_byte_identical",
        ):
            if row[key] is not True:
                raise AuthorityError(f"lean row exact boolean changed: {key}")
        derivation = exact_keys(
            row["h256_prefix_derivation"],
            H256_DERIVATION_KEYS,
            "lean H256 prefix derivation",
        )
        if (
            derivation["method"] != "EIGHT_SECTOR_RECEIVER_PREFIX_FROM_H256"
            or derivation["source_h"] != 256
            or derivation["derived_h"] != [16, 64]
            or derivation["all_derived_bytes_match_written_row"] is not True
            or derivation["sentinel_derived_matches_literal_and_reference"]
            is not (True if sentinel else None)
        ):
            raise AuthorityError("lean H256 prefix derivation evidence changed")

        comparator = row["source_action_comparator"]
        parent = row["five_component_parent_comparator"]
        literal: dict[str, Any] | None = None
        reference: dict[str, Any] | None = None
        if sentinel:
            comparator = exact_keys(
                comparator,
                SOURCE_ACTION_COMPARATOR_KEYS,
                "lean sentinel source-action comparator",
            )
            if (
                comparator["comparator"]
                != "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE"
                or comparator["literal_implementation_id"]
                != "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2"
                or comparator["reference_implementation_id"]
                != "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2"
                or comparator["literal_source_sha256"]
                != RUNTIME_V4_COMPONENT_SHA256["signed_pair_literal.py"]
                or comparator["reference_source_sha256"]
                != RUNTIME_V4_COMPONENT_SHA256["signed_pair_reference.py"]
                or comparator["all_action_and_A15_bytes_equal"] is not True
            ):
                raise AuthorityError("lean sentinel source-action comparator changed")
            literal = exact_keys(
                comparator["literal"],
                {"16", "64", "256"},
                "lean sentinel literal comparator",
            )
            reference = exact_keys(
                comparator["reference"],
                {"16", "64", "256"},
                "lean sentinel reference comparator",
            )
            parent = exact_keys(
                parent,
                PARENT_COMPARATOR_KEYS,
                "lean sentinel parent comparator",
            )
            if (
                parent["comparator"]
                != "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE"
                or parent["component_names"] != list(FIVE_COMPONENT_ORDER)
                or parent["all_five_component_bytes_equal"] is not True
            ):
                raise AuthorityError("lean sentinel parent comparator changed")
            base_components = exact_keys(
                parent["base_component_sha256"],
                FIVE_COMPONENT_NAMES,
                "lean sentinel base component hashes",
            )
            capture_components = exact_keys(
                parent["capture_component_sha256"],
                FIVE_COMPONENT_NAMES,
                "lean sentinel capture component hashes",
            )
            if base_components != capture_components or any(
                not is_sha(item) for item in base_components.values()
            ):
                raise AuthorityError("lean sentinel parent comparator failed")
        elif comparator is not None or parent is not None:
            raise AuthorityError("lean non-sentinel row contains dual-parity evidence")

        banks = exact_keys(row["banks"], {"16", "64", "256"}, "lean row banks")
        for h in HS:
            item = exact_keys(banks[str(h)], ROW_BANK_KEYS, "lean row bank")
            if (
                item["shape"] != [13, WIDTH, h]
                or item["dtype"] != "float32"
                or item["finite"] is not True
                or not is_sha(item["action_bytes_sha256"])
                or not is_sha(item["A15_zero_bytes_sha256"])
            ):
                raise AuthorityError("lean row bank schema changed")
            layers = item["layer_action_bytes_sha256"]
            if (
                type(layers) is not list
                or len(layers) != 13
                or any(not is_sha(item_hash) for item_hash in layers)
            ):
                raise AuthorityError("lean row bank layer hashes changed")
            if sentinel:
                if literal is None or reference is None:
                    raise AuthorityError("lean sentinel comparator evidence absent")
                for side in (literal, reference):
                    comparison = exact_keys(
                        side[str(h)],
                        {"action_sha256", "A15_zero_sha256"},
                        "lean sentinel source comparator H item",
                    )
                    if (
                        comparison["action_sha256"]
                        != item["action_bytes_sha256"]
                        or comparison["A15_zero_sha256"]
                        != item["A15_zero_bytes_sha256"]
                    ):
                        raise AuthorityError(
                            "lean sentinel source comparator/output mismatch"
                        )
                if literal[str(h)] != reference[str(h)]:
                    raise AuthorityError("lean sentinel literal/reference bytes differ")

        meters = exact_keys(row["exact_meters"], LEAN_ROW_METER_KEYS, "lean row meters")
        for key in LEAN_ROW_INTEGER_METER_KEYS:
            strict_int(meters[key], minimum=0)
        for key in ROW_FLOAT_METER_KEYS:
            if type(meters[key]) is not float:
                raise AuthorityError(f"lean row meter {key} must be a JSON float")
            _finite_nonnegative_number(meters[key], f"lean row meter {key}")
        if (
            meters["source_hook_capture_count"] != 13
            or meters["reference_source_hook_capture_count"]
            != (13 if sentinel else 0)
            or meters["meter_scope"]
            != "CAPTURE_LITERAL_ALL_ROWS_BASE_REFERENCE_SENTINELS"
        ):
            raise AuthorityError("lean row meter scope/sentinel ledger changed")
        for key in (
            "base_parent_flops",
            "base_parent_calls",
            "reference_source_hook_flops",
            "reference_source_hook_calls",
            "reference_workspace_flops",
            "reference_workspace_calls",
        ):
            if not sentinel and meters[key] != 0:
                raise AuthorityError(f"lean non-sentinel duplicate meter nonzero: {key}")
        observation = exact_keys(
            row["resource_observation"],
            RESOURCE_OBSERVATION_KEYS,
            "lean row resource observation",
        )
        if (
            observation["classification"] != "OBSERVATIONAL_DEPLOYMENT_ONLY"
            or observation["deployment_flop_cap"] != 2_199_023_255_552
            or type(observation["complete_flops_within_deployment_cap"]) is not bool
            or observation["peak_rss_bytes"] != meters["peak_rss_bytes"]
            or observation["wall_seconds"] != meters["wall_seconds"]
            or observation["residual_seconds"] != meters["residual_seconds"]
            or observation["accuracy_veto_allowed"] is not False
        ):
            raise AuthorityError("lean row resource observation changed")
    if len(set(mlp_ids)) != ROWS:
        raise AuthorityError("lean runtime-V4 mlp_id roster is not unique")

    final = exact_keys(receipt["complete_banks"], LEAN_FINAL_BANK_KEYS, "lean complete banks")
    if (
        strict_int(final["row_count"]) != ROWS
        or final["sentinel_ranks"] != list(SENTINEL_RANKS)
        or final["cross_rung_H16_prefix_byte_identical"] is not True
        or final["h256_prefix_derivation_complete_all_rows"] is not True
        or final["literal_reference_bytes_equal_sentinels"] is not True
        or final["base_capture_five_components_equal_sentinels"] is not True
        or final["resource_policy"]
        != "RESOURCE_AND_RESIDUAL_OBSERVATIONAL_DEPLOYMENT_ONLY"
    ):
        raise AuthorityError("lean complete-bank declarations changed")
    validate_ordered_ranks(final["ranks"])
    complete = exact_keys(final["banks"], {"16", "64", "256"}, "lean complete bank map")
    for h in HS:
        item = exact_keys(
            complete[str(h)], COMPLETE_BANK_ITEM_KEYS, "lean complete bank item"
        )
        canonical_relative_path(item["path"], f"lean H{h} bank path")
        canonical_relative_path(item["A15_zero_path"], f"lean H{h} A15 path")
        if (
            item["shape"] != [ROWS, 13, WIDTH, h]
            or item["dtype"] != "float32"
            or item["finite"] is not True
            or not is_sha(item["sha256"])
            or not is_sha(item["A15_zero_sha256"])
        ):
            raise AuthorityError("lean complete bank item schema changed")
    meter_vectors = exact_keys(
        final["exact_meters"], LEAN_FINAL_METER_KEYS, "lean meter vectors"
    )
    for name, vector in meter_vectors.items():
        if type(vector) is not list or len(vector) != ROWS:
            raise AuthorityError(f"lean meter vector length changed: {name}")
        for rank, meter in enumerate(vector):
            scalar_name = name.removesuffix("_by_rank")
            if name in LEAN_FINAL_INTEGER_METER_KEYS:
                strict_int(meter, minimum=0)
            else:
                if type(meter) is not float:
                    raise AuthorityError(
                        f"lean meter vector {name} must contain JSON floats"
                    )
                _finite_nonnegative_number(meter, f"lean meter vector {name}")
            if meter != rows[rank]["exact_meters"][scalar_name]:
                raise AuthorityError(f"lean meter vector/row scalar mismatch: {name}")
    consumption = exact_keys(
        final["capability_consumption"],
        {
            "capability_id",
            "issued_by",
            "issued_utc",
            "consumed_path",
            "consumed_sha256",
            "original_absent",
            "one_use_inode",
            "one_use",
            "no_retry",
        },
        "lean runtime-V4 capability consumption",
    )
    canonical_id(consumption["capability_id"], "lean generation capability_id")
    canonical_id(consumption["issued_by"], "lean generation capability issuer")
    canonical_utc(consumption["issued_utc"], "lean generation capability issued_utc")
    if (
        not is_sha(consumption["consumed_sha256"])
        or consumption["consumed_sha256"]
        != chain["single_use_capability_sha256"]
        or consumption["original_absent"] is not True
        or consumption["one_use"] is not True
        or consumption["no_retry"] is not True
    ):
        raise AuthorityError("lean runtime-V4 capability consumption changed")
    inode = exact_keys(
        consumption["one_use_inode"],
        {"device", "inode", "size", "initial_link_count", "consumed_link_count"},
        "lean runtime-V4 one-use inode",
    )
    for name in ("device", "inode", "size"):
        strict_int(inode[name], minimum=0)
    if inode["initial_link_count"] != 1 or inode["consumed_link_count"] != 1:
        raise AuthorityError("lean runtime-V4 same-inode consumption ledger changed")
    return receipt, mlp_ids


def validate_lean_runtime_v5_generation_receipt_object(
    value: Any,
) -> tuple[dict[str, Any], list[int]]:
    """Validate V5 control roots with the exact frozen V4 science schema.

    Runtime V5 deliberately emits the V4 generation-receipt and nested
    row/bank keysets.  First authenticate the V5-only control roots, then map
    only those roots in a detached compatibility copy and reuse the complete
    V4 scientific receipt validator.  The returned receipt is the original V5
    object; no scientific value is rewritten.
    """
    receipt = exact_keys(value, LEAN_RECEIPT_KEYS, "lean runtime-V5 receipt")
    if (
        receipt["schema"]
        != "public-q24-signed-pair-three-bank-generation-receipt-v4"
    ):
        raise AuthorityError("Runtime V5 generation receipt schema drifted from V4")
    chain = exact_keys(
        receipt["authority_chain"],
        RECEIPT_AUTHORITY_CHAIN_KEYS,
        "lean runtime-V5 receipt authority chain",
    )
    if (
        chain["callable_bundle_manifest_sha256"]
        != RUNTIME_V5_HASHES["callable_bundle_manifest_sha256"]
        or chain["execution_stack_manifest_sha256"]
        != RUNTIME_V5_HASHES["execution_stack_manifest_sha256"]
        or chain["independent_review_sha256"]
        != RUNTIME_V5_HASHES["independent_review_sha256"]
    ):
        raise AuthorityError("lean runtime-V5 authority chain changed")
    components = exact_keys(
        receipt["callable_component_sha256"],
        set(RUNTIME_V5_COMPONENT_SHA256),
        "lean runtime-V5 callable components",
    )
    if components != RUNTIME_V5_COMPONENT_SHA256:
        raise AuthorityError("lean runtime-V5 callable component root changed")
    expected_runtime_bindings = {
        "runner_sha256": RUNTIME_V5_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": RUNTIME_V5_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": RUNTIME_V5_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "independent_review_sha256": RUNTIME_V5_HASHES[
            "independent_review_sha256"
        ],
        "ordered_688_row_scope_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_row_scope_sha256"
        ],
        "ordered_688_record_sha256": ORDERED_SCOPE_BINDINGS[
            "ordered_688_record_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": RUNTIME_V5_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ],
    }
    release_bindings = exact_keys(
        receipt["release_bindings"],
        V4_RUNTIME_BINDING_KEYS,
        "lean runtime-V5 release bindings",
    )
    anchor_bindings = exact_keys(
        receipt["anchor_bindings"],
        V4_RUNTIME_BINDING_KEYS,
        "lean runtime-V5 anchor bindings",
    )
    if (
        release_bindings != expected_runtime_bindings
        or anchor_bindings != expected_runtime_bindings
    ):
        raise AuthorityError("lean runtime-V5 receipt control bindings changed")

    compatibility = copy.deepcopy(receipt)
    compatibility["authority_chain"]["independent_review_sha256"] = (
        RUNTIME_V4_HASHES["independent_review_sha256"]
    )
    for name in ("release_bindings", "anchor_bindings"):
        compatibility[name]["runner_sha256"] = RUNTIME_V4_HASHES[
            "runner_sha256"
        ]
        compatibility[name]["independent_review_sha256"] = RUNTIME_V4_HASHES[
            "independent_review_sha256"
        ]
    _, mlp_ids = validate_lean_runtime_v4_generation_receipt_object(compatibility)
    return receipt, mlp_ids




def validate_generation_receipt_object(
    value: Any,
) -> tuple[dict[str, Any], list[int]]:
    """Dispatch one frozen, exact generation-receipt compatibility profile."""
    if GENERATION_RECEIPT_PROFILE == "EXACT_LEAN_RUNTIME_V5":
        return validate_lean_runtime_v5_generation_receipt_object(value)
    raise AuthorityError("unsupported frozen generation receipt profile")


def validate_generation_provenance(
    authority: ExecutionAuthority,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    bank_bindings = authority.anchor["bank_bindings"]
    identities = authority.external_files
    raw_authority, source_authority = read_canonical_json(
        identities["bank_source_authority"], "pre-generation source authority"
    )
    raw_authority_review, source_review = read_canonical_json(
        identities["bank_source_authority_review"],
        "pre-generation source authority review",
    )
    raw_seal, seal = read_canonical_json(
        identities["bank_post_generation_seal"], "post-generation seal"
    )
    raw_seal_review, seal_review = read_canonical_json(
        identities["bank_post_generation_seal_review"],
        "post-generation seal independent review",
    )
    raw_receipt, receipt = read_canonical_json(
        identities["bank_generation_receipt"], "three-bank generation receipt"
    )

    authority_sha = bytes_sha256(raw_authority)
    authority_review_sha = bytes_sha256(raw_authority_review)
    seal_sha = bytes_sha256(raw_seal)
    seal_review_sha = bytes_sha256(raw_seal_review)
    receipt_sha = bytes_sha256(raw_receipt)
    if (
        authority_sha != bank_bindings["source_authority_sha256"]
        or authority_review_sha != bank_bindings["source_authority_review_sha256"]
        or seal_sha != bank_bindings["post_generation_seal_sha256"]
        or seal_review_sha != bank_bindings["post_generation_seal_review_sha256"]
        or receipt_sha != bank_bindings["generation_receipt_sha256"]
    ):
        raise AuthorityError("external generation provenance differs from execution anchor")

    checked_authority = validate_lean_v5_source_authority_object(source_authority)
    validate_lean_v5_source_authority_review_object(
        source_review,
        authority_sha256=authority_sha,
        authority=checked_authority,
    )
    checked_seal = validate_lean_v5_post_generation_seal_object(seal)
    validate_lean_v5_post_generation_seal_review_object(
        seal_review, seal_sha256=seal_sha, seal=checked_seal
    )
    checked_receipt, mlp_ids = validate_generation_receipt_object(receipt)

    if not (
        checked_authority["issued_utc"] <= source_review["issued_utc"]
        <= checked_seal["issued_utc"] <= seal_review["issued_utc"]
    ):
        raise AuthorityError("generation provenance chronology changed")

    if (
        checked_seal["pre_generation_source_authority_sha256"] != authority_sha
        or checked_seal["pre_generation_source_authority_review_sha256"]
        != authority_review_sha
        or checked_seal["pre_generation_v4_terminal_successor_record_sha256"]
        != checked_authority["source_bindings"][
            "v4_terminal_successor_record_sha256"
        ]
        or checked_seal["v4_terminal_failure_bindings"]
        != V4_TERMINAL_FAILURE_BINDINGS
        or checked_seal["independent_incident_root_cause_sha256"]
        != checked_authority["source_bindings"][
            "independent_incident_root_cause_sha256"
        ]
        or checked_seal["flopscope_reset_verification_sha256"]
        != checked_authority["source_bindings"][
            "flopscope_reset_verification_sha256"
        ]
        or checked_seal["manifest_compatibility_scan_receipt_sha256"]
        != checked_authority["source_bindings"][
            "manifest_compatibility_scan_receipt_sha256"
        ]
    ):
        raise AuthorityError("post-generation seal is not rooted in reviewed pre-generation authority")
    source = checked_authority["source_bindings"]
    transaction = checked_seal["generation_transaction_bindings"]
    receipt_chain = checked_receipt["authority_chain"]
    if (
        transaction["generation_execution_anchor_sha256"]
        != source["generation_execution_anchor_sha256"]
        or transaction["generation_single_use_capability_sha256"]
        != source["generation_single_use_capability_sha256"]
        or transaction["generation_final_release_sha256"]
        != source["generation_final_release_sha256"]
        or transaction["generation_execution_anchor_sha256"]
        != receipt_chain["authority_anchor_sha256"]
        or transaction["generation_final_release_sha256"]
        != receipt_chain["final_release_sha256"]
        or transaction["generation_single_use_capability_sha256"]
        != receipt_chain["single_use_capability_sha256"]
        or transaction["generation_claim_sha256"] != checked_receipt["claim_sha256"]
        or transaction["consumed_capability_sha256"]
        != checked_receipt["runtime_token_receipt"]["consumed_capability_sha256"]
    ):
        raise AuthorityError("actual generation escaped pre-generation allowlist")
    if transaction["generation_receipt_sha256"] != receipt_sha:
        raise AuthorityError("post-generation seal receipt binding mismatch")
    if (
        checked_seal["runtime_v5_static_bindings"] != RUNTIME_V5_HASHES
        or checked_seal["lean_generation_law"]
        != checked_authority["generation_law"]
    ):
        raise AuthorityError("post-generation seal escaped lean runtime/source law")
    scope = checked_authority["row_scope_bindings"]
    if checked_seal["ordered_scope_bindings"] != scope:
        raise AuthorityError("post-generation seal row scope differs from pre-generation authority")
    if (
        checked_receipt["ordered_688_row_scope_sha256"]
        != scope["ordered_688_row_scope_sha256"]
        or checked_receipt["ordered_688_record_sha256"]
        != scope["ordered_688_record_sha256"]
        or scope["ordered_688_record_sha256"]
        != bank_bindings["ordered_688_record_sha256"]
    ):
        raise AuthorityError("ordered 688-record binding changed across authority chain")

    direct_anchor_crosswalk = {
        "source_authority_sha256": authority_sha,
        "source_authority_review_sha256": authority_review_sha,
        "pre_generation_v4_terminal_successor_record_sha256": source[
            "v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_lineage_sha256": source[
            "v4_terminal_failure_lineage_sha256"
        ],
        "independent_incident_root_cause_sha256": source[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": source[
            "flopscope_reset_verification_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": source[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "v4_exclusive_claim_sha256": source["v4_exclusive_claim_sha256"],
        "v4_terminal_failure_sha256": source["v4_terminal_failure_sha256"],
        "v4_consumed_capability_sha256": source[
            "v4_consumed_capability_sha256"
        ],
        "lean_runtime_v5_package_manifest_sha256": RUNTIME_V5_HASHES[
            "package_manifest_sha256"
        ],
        "lean_runtime_v5_independent_review_sha256": RUNTIME_V5_HASHES[
            "independent_review_sha256"
        ],
        "generation_execution_anchor_sha256": transaction[
            "generation_execution_anchor_sha256"
        ],
        "generation_final_release_sha256": transaction[
            "generation_final_release_sha256"
        ],
        "generation_single_use_capability_sha256": transaction[
            "generation_single_use_capability_sha256"
        ],
        "generation_claim_sha256": transaction["generation_claim_sha256"],
        "generation_receipt_sha256": receipt_sha,
        "post_generation_seal_sha256": seal_sha,
        "post_generation_seal_review_sha256": seal_review_sha,
        "ordered_688_record_sha256": scope["ordered_688_record_sha256"],
    }
    for key, expected in direct_anchor_crosswalk.items():
        if bank_bindings[key] != expected:
            raise AuthorityError(f"nested anchor generation binding mismatch: {key}")

    receipt_banks = checked_receipt["complete_banks"]["banks"]
    sealed_banks = checked_seal["bank_bindings"]
    for h in HS:
        key = str(h)
        if (
            receipt_banks[key]["sha256"]
            != sealed_banks["complete_bank_sha256"][key]
            or receipt_banks[key]["sha256"]
            != bank_bindings["complete_bank_sha256"][key]
            or receipt_banks[key]["A15_zero_sha256"]
            != sealed_banks["A15_zero_sha256"][key]
            or receipt_banks[key]["A15_zero_sha256"]
            != bank_bindings["A15_zero_sha256"][key]
        ):
            raise AuthorityError(f"H{h} complete-bank chain binding mismatch")
    provenance = {
        "pre_generation_source_authority_sha256": authority_sha,
        "pre_generation_source_authority_review_sha256": authority_review_sha,
        "pre_generation_v4_terminal_successor_record_sha256": source[
            "v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_bindings": copy.deepcopy(
            V4_TERMINAL_FAILURE_BINDINGS
        ),
        "independent_incident_root_cause_sha256": source[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": source[
            "flopscope_reset_verification_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": source[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "runtime_v5_static_bindings": copy.deepcopy(RUNTIME_V5_HASHES),
        "lean_generation_law": copy.deepcopy(checked_authority["generation_law"]),
        "post_generation_seal_sha256": seal_sha,
        "post_generation_seal_review_sha256": seal_review_sha,
        "generation_transaction_bindings": copy.deepcopy(transaction),
        "generation_receipt_sha256": receipt_sha,
        "sealer_access_accounting": copy.deepcopy(checked_seal["access_accounting"]),
        "target_freedom_evidence_scope": checked_seal["limitations"][
            "generation_target_freedom_basis"
        ],
        "authentication": (
            "INDEPENDENT_POST_GENERATION_V5_REVIEW_BOUND_TO_SEPARATE_"
            "PRE_GENERATION_SOURCE_AUTHORITY_V5_V4_TERMINAL_INCIDENT_AND_"
            "EXACT_LEAN_RUNTIME_V5_RESET_ROOTS"
        ),
    }
    return checked_receipt, mlp_ids, provenance


def _read_exact_descriptor(descriptor: int, count: int, label: str) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        block = os.read(descriptor, remaining)
        if not block:
            raise AuthorityError(f"short complete-bank read: {label}")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def recompute_complete_bank_evidence(
    *,
    bank_identities: dict[int, PathIdentity],
    zero_identities: dict[int, PathIdentity],
    row_receipts: list[dict[str, Any]],
    claimed_complete_sha256: dict[str, str],
    claimed_zero_sha256: dict[str, str],
    rows: int,
    width: int,
    layers: int,
    hs: tuple[int, ...],
    sectors: int,
) -> dict[str, Any]:
    """Recompute every digest and the lean runtime-V4 eight-sector prefix law."""
    import numpy as np  # Post-claim integrity dependency; no target is decoded.

    if hs != (16, 64, 256) or sectors != 8:
        raise AuthorityError("lean runtime-V4 sector-prefix geometry changed")

    descriptors = {h: _open_identity(bank_identities[h]) for h in hs}
    complete_digests = {h: hashlib.sha256() for h in hs}
    bytes_read = {h: 0 for h in hs}
    row_hashes_verified = 0
    layer_hashes_verified = 0
    prefix_comparisons = 0
    try:
        for rank in range(rows):
            row_digests = {h: hashlib.sha256() for h in hs}
            for layer_slot in range(layers):
                payloads: dict[int, bytes] = {}
                for h in hs:
                    count = width * h * 4
                    payload = _read_exact_descriptor(
                        descriptors[h], count, f"H{h}/rank{rank}/slot{layer_slot}"
                    )
                    payloads[h] = payload
                    complete_digests[h].update(payload)
                    row_digests[h].update(payload)
                    bytes_read[h] += len(payload)
                    observed_layer = bytes_sha256(payload)
                    claimed_layer = row_receipts[rank]["banks"][str(h)][
                        "layer_action_bytes_sha256"
                    ][layer_slot]
                    if observed_layer != claimed_layer:
                        raise AuthorityError(
                            f"direct layer hash mismatch H{h}/rank{rank}/slot{layer_slot}"
                        )
                    floats = np.frombuffer(payload, dtype=np.dtype("<f4"))
                    if not bool(np.isfinite(floats).all()):
                        raise AuthorityError(
                            f"nonfinite complete-bank bytes H{h}/rank{rank}/slot{layer_slot}"
                        )
                    layer_hashes_verified += 1
                sector_bytes = {
                    h: np.frombuffer(payloads[h], dtype=np.uint8).reshape(
                        width, sectors, (h // sectors) * 4
                    )
                    for h in hs
                }
                comparisons = (
                    (16, 64, sector_bytes[16], sector_bytes[64][:, :, : 2 * 4]),
                    (16, 256, sector_bytes[16], sector_bytes[256][:, :, : 2 * 4]),
                    (64, 256, sector_bytes[64], sector_bytes[256][:, :, : 8 * 4]),
                )
                for narrow_h, wide_h, narrow, wide_prefix in comparisons:
                    if not bool(np.array_equal(narrow, wide_prefix)):
                        raise AuthorityError(
                            "direct eight-sector prefix mismatch "
                            f"H{narrow_h}->H{wide_h}/rank{rank}/slot{layer_slot}"
                        )
                    prefix_comparisons += 1
            for h in hs:
                observed_row = row_digests[h].hexdigest()
                claimed_row = row_receipts[rank]["banks"][str(h)][
                    "action_bytes_sha256"
                ]
                if observed_row != claimed_row:
                    raise AuthorityError(f"direct row hash mismatch H{h}/rank{rank}")
                row_hashes_verified += 1
        for h in hs:
            if os.read(descriptors[h], 1):
                raise AuthorityError(f"complete bank H{h} has trailing bytes")
            observed_complete = complete_digests[h].hexdigest()
            if observed_complete != claimed_complete_sha256[str(h)]:
                raise AuthorityError(f"direct complete-file hash mismatch H{h}")
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    zero_bytes_read: dict[int, int] = {}
    for h in hs:
        descriptor = _open_identity(zero_identities[h])
        digest = hashlib.sha256()
        count = 0
        try:
            remaining = zero_identities[h].size
            while remaining:
                block = os.read(descriptor, min(1 << 20, remaining))
                if not block:
                    raise AuthorityError(f"short A15 read H{h}")
                if any(block):
                    raise AuthorityError(f"A15 contains a nonzero byte H{h}")
                digest.update(block)
                count += len(block)
                remaining -= len(block)
            if os.read(descriptor, 1):
                raise AuthorityError(f"A15 H{h} has trailing bytes")
        finally:
            os.close(descriptor)
        observed_zero = digest.hexdigest()
        if observed_zero != claimed_zero_sha256[str(h)]:
            raise AuthorityError(f"direct A15 hash mismatch H{h}")
        for rank in range(rows):
            if (
                row_receipts[rank]["banks"][str(h)]["A15_zero_bytes_sha256"]
                != observed_zero
            ):
                raise AuthorityError(f"row A15 hash mismatch H{h}/rank{rank}")
        zero_bytes_read[h] = count
    return {
        "row_hashes_recomputed_from_complete_bytes": row_hashes_verified,
        "layer_hashes_recomputed_from_complete_bytes": layer_hashes_verified,
        "cross_rung_prefix_comparisons_from_complete_bytes": prefix_comparisons,
        "complete_bank_bytes_read": {str(h): bytes_read[h] for h in hs},
        "A15_bytes_read": {str(h): zero_bytes_read[h] for h in hs},
        "all_finite_directly_checked": True,
        "all_prefixes_byte_identical_directly_checked": True,
        "prefix_law": (
            "8_SECTORS__H16_FIRST2_PER_SECTOR__H64_FIRST8_PER_SECTOR__"
            "H256_ALL32_PER_SECTOR"
        ),
    }


def verify_complete_bank_bytes(
    authority: ExecutionAuthority,
    receipt: dict[str, Any],
    mlp_ids: list[int],
    provenance: dict[str, Any],
) -> BankAuthority:
    final = receipt["complete_banks"]
    complete = final["banks"]
    bank_identities: dict[int, PathIdentity] = {}
    zero_identities: dict[int, PathIdentity] = {}
    for h in HS:
        item = complete[str(h)]
        bank_identities[h] = secure_relative_file(
            authority.bank_root, item["path"], expected_size=EXACT_BANK_BYTES[h]
        )
        zero_identities[h] = secure_relative_file(
            authority.bank_root,
            item["A15_zero_path"],
            expected_size=EXACT_ZERO_BYTES[h],
        )
    integrity = recompute_complete_bank_evidence(
        bank_identities=bank_identities,
        zero_identities=zero_identities,
        row_receipts=receipt["row_receipts"],
        claimed_complete_sha256=authority.anchor["bank_bindings"][
            "complete_bank_sha256"
        ],
        claimed_zero_sha256=authority.anchor["bank_bindings"]["A15_zero_sha256"],
        rows=ROWS,
        width=WIDTH,
        layers=13,
        hs=HS,
        sectors=8,
    )
    integrity["receipt_declared_prefix_parity"] = (
        final["cross_rung_H16_prefix_byte_identical"] is True
    )
    integrity["authenticated_post_generation_provenance"] = copy.deepcopy(provenance)
    return BankAuthority(
        receipt_path=authority.external_files["bank_generation_receipt"].resolved,
        receipt_sha256=authority.anchor["bank_bindings"]["generation_receipt_sha256"],
        ordered_record_sha256=authority.anchor["bank_bindings"][
            "ordered_688_record_sha256"
        ],
        mlp_ids=mlp_ids,
        bank_paths={h: bank_identities[h].resolved for h in HS},
        zero_paths={h: zero_identities[h].resolved for h in HS},
        resource_observations=copy.deepcopy(final["exact_meters"]),
        novelty_diagnostics={
            "status": "DIAGNOSTIC_ONLY_NOT_REQUESTED__ALL_H_REMAIN_ELIGIBLE",
            "can_veto_accuracy_or_H": False,
            "authenticated_bank_generation": copy.deepcopy(provenance),
        },
        integrity_receipt=integrity,
    )


def _hash_fixed_input(path: Path, expected: str, label: str) -> tuple[PathIdentity, str]:
    identity = secure_file(path)
    observed = hash_identity(identity)
    if observed != expected:
        raise AuthorityError(f"frozen {label} bytes changed")
    return identity, observed


def audited_full_preflight(
    authority: ExecutionAuthority, consumed_capability: PathIdentity
) -> tuple[BankAuthority, dict[str, Any]]:
    """Post-claim full audit.  It hashes bytes but decodes no target value."""
    if not claim_owned():
        raise AuthorityError("audited full preflight requires owned durable claim")
    started = time.perf_counter()
    dynamic_inputs = {
        "J_source_sha256": J_SOURCE,
        "QREF_source_sha256": QREF_SOURCE,
        "raw_sufficient_stats_sha256": RAW_STATS,
        "feature_manifest_sha256": FEATURE_MANIFEST,
        "target_bundle_sha256": TARGETS,
    }
    dynamic_identities: dict[str, PathIdentity] = {}
    for key, path in dynamic_inputs.items():
        identity, _ = _hash_fixed_input(
            path, authority.anchor["dynamic_input_bindings"][key], key
        )
        dynamic_identities[key] = identity
    science_identity, _ = _hash_fixed_input(
        SCIENCE_RUNNER,
        authority.anchor["package_bindings"]["v1_science_runner_sha256"],
        "reviewed V1 science runner",
    )

    predecessor_sizes: dict[str, int] = {}
    for path, expected in PREDECESSOR_HASHES.items():
        identity, _ = _hash_fixed_input(path, expected, str(path.relative_to(P2)))
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    for path, expected in RUNTIME_V5_PATH_HASHES.items():
        identity, _ = _hash_fixed_input(
            path, expected, f"selected lean runtime-V5 root {path.relative_to(P2)}"
        )
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    for path, expected in V5_AUTHORITY_BUILDER_PATH_HASHES.items():
        identity, _ = _hash_fixed_input(
            path,
            expected,
            f"selected V5 authority-builder root {path.relative_to(P2)}",
        )
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    # Runtime V4 is immutable terminal/no-retry lineage.  Only its frozen
    # public control/source roots are hashed; its forensic staging tree is not
    # listed here and is never descended or accepted as an input.
    for path, expected in RUNTIME_V4_PATH_HASHES.items():
        identity, _ = _hash_fixed_input(
            path, expected, f"terminal runtime-V4 lineage {path.relative_to(P2)}"
        )
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    # V3 is immutable failed lineage: it terminated before row 0, consumed its
    # capability, published no bundle, and is not an acceptable receipt source.
    for path, expected in RUNTIME_V3_PATH_HASHES.items():
        identity, _ = _hash_fixed_input(
            path, expected, f"terminal runtime-V3 lineage {path.relative_to(P2)}"
        )
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    # Runtime V2 is an immutable, unconsumed fallback root.  Hashing it does
    # not make its receipt acceptable under the frozen V5 selector.
    for path, expected in RUNTIME_V2_PATH_HASHES.items():
        identity, _ = _hash_fixed_input(
            path, expected, f"unconsumed runtime-V2 fallback root {path.relative_to(P2)}"
        )
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    for path, expected in SOURCE_V1_PATH_HASHES.items():
        identity, _ = _hash_fixed_input(
            path, expected, f"three-bank source root {path.relative_to(P2)}"
        )
        predecessor_sizes[str(path.relative_to(P2))] = identity.size
    v1_extra = {
        "public_q24_signed_pair_nested_accuracy_oof_v1/PACKAGE_MANIFEST.json": (
            V1 / "PACKAGE_MANIFEST.json",
            V1_REVIEWED_SHA256["manifest"],
        ),
        "public_q24_signed_pair_nested_accuracy_oof_v1/FREEZE_VERIFICATION.json": (
            V1 / "FREEZE_VERIFICATION.json",
            V1_REVIEWED_SHA256["freeze_verification"],
        ),
        "public_q24_signed_pair_nested_accuracy_oof_v1/STATIC_VERIFICATION.json": (
            V1 / "STATIC_VERIFICATION.json",
            V1_REVIEWED_SHA256["static_verification"],
        ),
    }
    for label, (path, expected) in v1_extra.items():
        identity, _ = _hash_fixed_input(path, expected, label)
        predecessor_sizes[label] = identity.size

    receipt, mlp_ids, provenance = validate_generation_provenance(authority)
    bank_authority = verify_complete_bank_bytes(
        authority, receipt, mlp_ids, provenance
    )
    integrity = bank_authority.integrity_receipt
    external_sizes = {
        name: identity.size for name, identity in authority.external_files.items()
    }
    full = {
        "schema": "public-q24-signed-pair-nested-accuracy-full-preflight-v3",
        "status": "PASS_POST_CLAIM_FULL_AUDIT__ZERO_TARGET_VALUES_DECODED",
        "claim_sha256": hash_identity(secure_file(CLAIM)),
        "consumed_capability_sha256": hash_identity(consumed_capability),
        "execution_anchor_sha256": bytes_sha256(authority.anchor_raw),
        "final_release_sha256": bytes_sha256(authority.release_raw),
        "predecessor_bytes_hashed": predecessor_sizes,
        "external_authority_bytes_opened_and_hashed": external_sizes,
        "access_truth": {
            "metadata_only_preflight_manifest_bytes_opened": 0,
            "metadata_only_preflight_dynamic_source_bytes_opened": 0,
            "metadata_only_preflight_bank_bytes_opened": 0,
            "metadata_only_preflight_target_bundle_bytes_opened": 0,
            "post_claim_manifest_files_opened_and_hashed": {
                "target_blind_feature_store_manifest": dynamic_identities[
                    "feature_manifest_sha256"
                ].size,
                "signed_pair_v2_package_manifest": secure_file(
                    ACTION_V2 / "PACKAGE_MANIFEST.json"
                ).size,
                "v1_accuracy_package_manifest": secure_file(
                    V1 / "PACKAGE_MANIFEST.json"
                ).size,
            },
            "post_claim_dynamic_source_files_opened_and_hashed": {
                "reviewed_V1_science_runner": science_identity.size,
                "J_source": dynamic_identities["J_source_sha256"].size,
                "QREF_source": dynamic_identities["QREF_source_sha256"].size,
            },
            "post_claim_raw_sufficient_stats_bytes_opened_and_hashed": (
                dynamic_identities["raw_sufficient_stats_sha256"].size
            ),
            "post_claim_target_bundle_bytes_opened_and_hashed": dynamic_identities[
                "target_bundle_sha256"
            ].size,
            "post_claim_target_values_decoded": 0,
            "post_claim_row_or_weight_values_opened": 0,
            "post_claim_complete_bank_bytes_opened": integrity[
                "complete_bank_bytes_read"
            ],
            "post_claim_A15_bytes_opened": integrity["A15_bytes_read"],
            "pyarrow_imported": "pyarrow" in sys.modules,
        },
        "generation_provenance": provenance,
        "complete_bank_integrity": integrity,
        "resources": observation("post_claim_full_preflight", started),
        "decision": "PROCEED_TO_FROZEN_SCIENCE_IMPORT_AND_TARGET_DECODE_ONCE",
    }
    if full["access_truth"]["pyarrow_imported"] is not False:
        raise AuthorityError("pyarrow imported before target-decode boundary")
    return bank_authority, full


def load_module_after_claim(name: str, identity: PathIdentity):
    if not claim_owned() or not FULL_PREFLIGHT.is_file():
        raise AuthorityError("dynamic import requires owned claim and durable full preflight")
    current = secure_file(identity.unresolved, expected_size=identity.size)
    if current.device != identity.device or current.inode != identity.inode:
        raise AuthorityError("dynamic module identity changed after full preflight")
    specification = importlib.util.spec_from_file_location(name, identity.resolved)
    if specification is None or specification.loader is None:
        raise ImportError(identity.resolved)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def claim_owned() -> bool:
    if ACTIVE_CLAIM_NONCE is None:
        return False
    try:
        identity = secure_file(CLAIM)
        _, value = read_canonical_json(identity, "run claim")
    except Exception:
        return False
    return (
        value.get("owner_pid") == os.getpid()
        and value.get("owner_nonce") == ACTIVE_CLAIM_NONCE
    )


def record_terminal_failure(error: BaseException) -> bool:
    if not claim_owned() or FAILURE.exists() or BUNDLE.exists():
        return False
    value = {
        "schema": "public-q24-signed-pair-nested-accuracy-terminal-failure-v3",
        "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
        "claim_sha256": hash_identity(secure_file(CLAIM)),
        "full_preflight_present": FULL_PREFLIGHT.exists(),
        "full_preflight_sha256": (
            hash_identity(secure_file(FULL_PREFLIGHT)) if FULL_PREFLIGHT.exists() else None
        ),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "private_partial_bundle": str(ACTIVE_TEMP_BUNDLE) if ACTIVE_TEMP_BUNDLE else None,
        "public_bundle_present": BUNDLE.exists(),
        "decision": "NO_RETRY__ATTRIBUTE_FAILURE_BEFORE_SUCCESSOR",
    }
    write_exclusive_durable(FAILURE, canonical_json_bytes(value))
    return True


def run_frozen_science(
    science: Any,
    authority: ExecutionAuthority,
    consumed_capability: PathIdentity,
    bank_authority: BankAuthority,
) -> None:
    """Run the reviewed V1 nested-OOF science only after the V3 firewall."""
    global ACTIVE_TEMP_BUNDLE
    np = science.np
    torch = science.torch

    j_identity, _ = _hash_fixed_input(
        J_SOURCE,
        authority.anchor["dynamic_input_bindings"]["J_source_sha256"],
        "J source before import",
    )
    return _run_frozen_science_body(
        science,
        authority,
        consumed_capability,
        bank_authority,
        j_identity,
    )


def execute(release_path: Path, capability_path: Path) -> None:
    global ACTIVE_CLAIM_NONCE
    if GENERATION_RECEIPT_PROFILE != "EXACT_LEAN_RUNTIME_V5":
        raise AuthorityError(
            "nested accuracy V4 is held before transaction inspection: "
            "fresh Runtime V5 live roots are not bound"
        )
    if CLAIM.exists() or FAILURE.exists() or FULL_PREFLIGHT.exists() or BUNDLE.exists():
        raise FileExistsError("nested accuracy V3 transaction already claimed or completed")

    # Pre-claim reads are restricted to exact authority and static package bytes.
    # External generation/data paths are lstat-bound only; no manifest, dynamic
    # source, bank, row, weight, or target byte is opened here.
    authority = validate_execution_authorities(release_path, capability_path)
    owner_nonce = hashlib.sha256(
        os.getpid().to_bytes(8, "big", signed=False) + os.urandom(32)
    ).hexdigest()
    claim = {
        "schema": "public-q24-signed-pair-nested-accuracy-run-claim-v3",
        "status": (
            "CLAIMED_BEFORE_ANY_MANIFEST_DYNAMIC_SOURCE_BANK_ROW_WEIGHT_OR_TARGET_BYTES"
            "__NO_RETRY"
        ),
        "bindings": {
            "final_release_sha256": bytes_sha256(authority.release_raw),
            "execution_anchor_sha256": bytes_sha256(authority.anchor_raw),
            "capability_sha256": bytes_sha256(authority.capability_raw),
            "v1_science_runner_sha256": V1_REVIEWED_SHA256["science_runner"],
            "v1_hold_receipt_sha256": V1_REVIEWED_SHA256["hold_receipt"],
            "bank_generation_receipt_sha256": authority.anchor["bank_bindings"][
                "generation_receipt_sha256"
            ],
            "ordered_688_record_sha256": authority.anchor["bank_bindings"][
                "ordered_688_record_sha256"
            ],
            "target_bundle_sha256": DYNAMIC_HASHES["target_bundle_sha256"],
            "outer_fold_assignment_uint8_sha256": DYNAMIC_HASHES[
                "outer_fold_assignment_uint8_sha256"
            ],
        },
        "capability_input_identity": authority.capability_identity.claim_record(),
        "external_input_metadata_only_identities": {
            name: identity.claim_record()
            for name, identity in authority.external_files.items()
        },
        "bank_root_metadata_only_identity": authority.bank_root.claim_record(),
        "preclaim_access_truth": authority.preclaim_access,
        "capability_consumed_at_claim_write": False,
        "owner_pid": os.getpid(),
        "owner_nonce": owner_nonce,
    }
    write_exclusive_durable(CLAIM, canonical_json_bytes(claim))
    ACTIVE_CLAIM_NONCE = owner_nonce
    consumed_capability = consume_capability_unique(
        authority.capability_identity,
        authority.capability_raw,
        authority.capability["capability_id"],
    )
    bank_authority, full_preflight = audited_full_preflight(
        authority, consumed_capability
    )
    full_preflight["capability_consumption"] = {
        "input_identity": authority.capability_identity.claim_record(),
        "consumed_identity": consumed_capability.claim_record(),
        "consumed_sha256": hash_identity(consumed_capability),
        "input_had_exactly_one_link": authority.capability_identity.links == 1,
        "same_inode_hard_link_then_unlink": True,
        "copied_inode_created_with_O_EXCL": False,
        "inode_preserved": (
            consumed_capability.device == authority.capability_identity.device
            and consumed_capability.inode == authority.capability_identity.inode
        ),
        "original_absent": (
            not authority.capability_identity.unresolved.exists()
            and not authority.capability_identity.unresolved.is_symlink()
        ),
        "one_use": True,
        "retry": False,
    }
    write_exclusive_durable(FULL_PREFLIGHT, canonical_json_bytes(full_preflight))

    science_identity, _ = _hash_fixed_input(
        SCIENCE_RUNNER,
        V1_REVIEWED_SHA256["science_runner"],
        "reviewed V1 science runner before import",
    )
    science = load_module_after_claim(
        "signed_pair_nested_accuracy_reviewed_v1_science", science_identity
    )
    run_frozen_science(
        science, authority, consumed_capability, bank_authority
    )


def _metadata_path_state(path: Path) -> dict[str, Any]:
    """Return lstat-only state.  This function never opens or hashes bytes."""
    try:
        metadata = _lstat_without_symlink_components(path)
    except Exception as error:
        return {"status": f"HOLD:{type(error).__name__}:{error}"}
    return {
        "status": "REGULAR_ONE_LINK"
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        else "HOLD_NOT_REGULAR_UNIQUE",
        "size_from_lstat": metadata.st_size,
        "links_from_lstat": metadata.st_nlink,
    }


def metadata_preflight_snapshot(release_path: Path | None) -> dict[str, Any]:
    """Lstat-only preflight: zero artifact, manifest, source, or data reads."""
    release_state = (
        {"status": "HOLD_NO_RELEASE_PATH"}
        if release_path is None
        else _metadata_path_state(release_path)
    )
    artifact_states = {
        "claim": "ABSENT" if not CLAIM.exists() else "PRESENT",
        "full_preflight": "ABSENT" if not FULL_PREFLIGHT.exists() else "PRESENT",
        "terminal_failure": "ABSENT" if not FAILURE.exists() else "PRESENT",
        "output_bundle": "ABSENT" if not BUNDLE.exists() else "PRESENT",
    }
    ready = (
        release_state.get("status") == "REGULAR_ONE_LINK"
        and all(value == "ABSENT" for value in artifact_states.values())
    )
    return {
        "schema": "public-q24-signed-pair-nested-accuracy-metadata-preflight-v3",
        "status": "READY_FOR_EXACT_AUTHORITY_VALIDATION_AND_ONE_SHOT_CLAIM"
        if ready
        else "HOLD_METADATA_ONLY",
        "release_lstat_only": release_state,
        "transaction_artifact_lstat_only": artifact_states,
        "access_truth": {
            "V3_artifact_bytes_opened_or_hashed": 0,
            "release_bytes_opened_or_hashed": 0,
            "manifest_bytes_opened_or_hashed": 0,
            "dynamic_source_bytes_opened_or_hashed": 0,
            "bank_bytes_opened_or_hashed": 0,
            "row_or_weight_bytes_opened": 0,
            "target_bundle_bytes_opened_or_hashed": 0,
            "target_values_decoded": 0,
        },
        "runtime_metadata": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy_imported": "numpy" in sys.modules,
            "torch_imported": "torch" in sys.modules,
            "pyarrow_imported": "pyarrow" in sys.modules,
        },
        "next_boundary": (
            "execute validates exact acyclic authority bytes, claims once, consumes the "
            "capability, then performs the separately audited full preflight"
        ),
    }


def preflight(release_path: Path | None) -> None:
    print(json.dumps(metadata_preflight_snapshot(release_path), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--capability", type=Path)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.release)
        return
    if args.release is None or args.capability is None:
        raise SystemExit("--execute requires --release and --capability")
    try:
        execute(args.release, args.capability)
    except BaseException as error:
        record_terminal_failure(error)
        raise


def _run_frozen_science_body(
    science: Any,
    authority: ExecutionAuthority,
    consumed_capability: PathIdentity,
    bank_authority: BankAuthority,
    j_identity: PathIdentity,
) -> None:
    global ACTIVE_TEMP_BUNDLE
    np = science.np
    torch = science.torch
    qref_identity, _ = _hash_fixed_input(
        QREF_SOURCE,
        authority.anchor["dynamic_input_bindings"]["QREF_source_sha256"],
        "QREF source before import",
    )
    j = load_module_after_claim("signed_pair_accuracy_v2_bound_j", j_identity)
    qref = load_module_after_claim("signed_pair_accuracy_v2_bound_qref", qref_identity)
    records, native_storage = qref.verify_native_rows()
    record_ids = np.asarray([int(record["mlp_id"]) for record in records], dtype=np.int64)
    if not np.array_equal(
        record_ids, np.asarray(bank_authority.mlp_ids, dtype=np.int64)
    ):
        raise AuthorityError("authenticated three-bank and observer-fit identities differ")
    folds = qref.balanced_folds(records)
    observed_fold_hash = hashlib.sha256(
        np.ascontiguousarray(folds.astype("|u1", copy=False)).tobytes()
    ).hexdigest()
    if observed_fold_hash != DYNAMIC_HASHES["outer_fold_assignment_uint8_sha256"]:
        raise AuthorityError("outer fold bytes changed")
    dataset = qref.All688Dataset(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False)
                for rank in range(ROWS)
            ]
        ),
        dtype=np.float32,
    )
    if hosts.shape != (ROWS, DEPTH, WIDTH):
        raise AssertionError("host array schema changed")
    stats = np.load(RAW_STATS, allow_pickle=False)
    sums = stats["sum"]
    sums2 = stats["sum2"]
    if sums.dtype != np.float64 or sums.shape != (ROWS, DEPTH, 19) or sums2.shape != sums.shape:
        raise AssertionError("raw sufficient-stat schema changed")
    banks = science.open_bank_memmaps(bank_authority)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    owner_nonce = ACTIVE_CLAIM_NONCE
    if owner_nonce is None:
        raise AuthorityError("claim owner nonce missing")
    temporary = HERE / f".{BUNDLE.name}.{os.getpid()}.{owner_nonce[:16]}.tmp"
    temporary.mkdir(mode=0o700)
    ACTIVE_TEMP_BUNDLE = temporary
    saved: dict[str, Any] = {
        "schema": "public-q24-signed-pair-nested-accuracy-models-v3",
        "inner": {},
        "outer": {},
    }
    outer_receipts: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selection_hashes: dict[str, str] = {}
    for outer_fold in range(OUTER_FOLDS):
        outer_fit = np.flatnonzero(folds != outer_fold).astype(np.int32)
        outer_held = np.flatnonzero(folds == outer_fold).astype(np.int32)
        if len(outer_fit) != OUTER_FIT or len(outer_held) != OUTER_HELD:
            raise AssertionError("outer fold size changed")
        fit_targets = science.target_subset(qref, records, outer_fit)
        selection, inner_saved = science.run_inner_selection(
            j,
            dataset,
            outer_fold,
            outer_fit,
            fit_targets,
            hosts,
            sums,
            sums2,
            record_ids,
            banks,
        )
        if selection.get("candidate_H") != list(HS):
            raise AuthorityError("all three H did not enter frozen inner selection")
        selection_path = temporary / f"SELECTION_OUTER_FOLD_{outer_fold}.json"
        write_exclusive_durable(selection_path, canonical_json_bytes(selection))
        selection_identity = secure_file(selection_path)
        selection_hash = hash_identity(selection_identity)
        selection_hashes[str(outer_fold)] = selection_hash
        _, frozen_selection = read_canonical_json(
            selection_identity, "outer-fold H selection"
        )
        selected_h = strict_int(frozen_selection.get("selected_H"))
        if selected_h not in HS or hash_identity(selection_identity) != selection_hash:
            raise AuthorityError("durable H selection changed before held scoring")
        outer_receipt, outer_saved = science.train_and_score_outer(
            j,
            qref,
            records,
            dataset,
            outer_fold,
            outer_fit,
            outer_held,
            fit_targets,
            hosts,
            sums,
            sums2,
            selected_h,
            banks[selected_h],
        )
        outer_receipt["selection_receipt_sha256"] = selection_hash
        outer_receipts.append(outer_receipt)
        selected_rows.extend(outer_receipt["per_network"])
        saved["inner"][str(outer_fold)] = inner_saved
        saved["outer"][str(outer_fold)] = outer_saved

    selected_rows.sort(key=lambda row: int(row["role_rank"]))
    if [int(row["role_rank"]) for row in selected_rows] != list(range(ROWS)):
        raise AssertionError("final selected-H OOF coverage changed")
    final_summary = science.summarize_rows(selected_rows)
    fold_gates = []
    for receipt in outer_receipts:
        summary = receipt["summary"]
        fold_gates.append(
            {
                "outer_fold": receipt["outer_fold"],
                "selected_H": receipt["selected_H"],
                "pair_on_below_pair_zero": summary["pair_on_raw_mse"]
                < summary["pair_zero_raw_mse"],
                "pair_on_below_fold_J": summary["pair_on_raw_mse"]
                < summary["fold_J_raw_mse"],
            }
        )
    gates = {
        "finite_complete_688": True,
        "selected_pair_on_raw_strictly_below_1_2425668176231516e_8": float(
            final_summary["pair_on_raw_mse"]
        )
        < RAW_HURDLE,
        "pair_on_below_pair_zero_all_four_folds": all(
            row["pair_on_below_pair_zero"] for row in fold_gates
        ),
        "pair_on_below_fold_J_all_four_folds": all(
            row["pair_on_below_fold_J"] for row in fold_gates
        ),
        "strict_network_win_fraction_vs_fold_J_at_least_0_55": float(
            final_summary["strict_win_fraction_vs_fold_J"]
        )
        >= 0.55,
        "paired_z_vs_pair_zero_at_least_3": science.z_at_least(
            final_summary["paired_vs_pair_zero"], 3.0
        ),
        "aggregate_pair_on_below_fold_J": float(final_summary["pair_on_raw_mse"])
        < float(final_summary["fold_J_raw_mse"]),
    }
    passed = all(gates.values())

    models_path = temporary / "MODELS.pt"
    torch.save(saved, models_path)
    with models_path.open("rb") as handle:
        os.fsync(handle.fileno())
    result = {
        "schema": "public-q24-signed-pair-nested-accuracy-oof-result-v3",
        "status": "PASS_SIGNED_PAIR_RAW_HURDLE__NO_SUBMISSION_AUTHORITY"
        if passed
        else "FAIL_SIGNED_PAIR_FROZEN_NESTED_OOF__NO_RETRY",
        "access_declarations": {
            "role": "observer_fit",
            "target_columns_opened": ["mlp_id", "all_layer_means"],
            "target_values_decoded_only_after_claim_capability_and_full_preflight": True,
            "observer_selection_targets_opened": [],
            "names_opened": [],
            "scores_opened": [],
            "board_opened": False,
            "network": False,
            "package": False,
            "submission": False,
        },
        "bindings": {
            "claim_sha256": hash_identity(secure_file(CLAIM)),
            "full_preflight_sha256": hash_identity(secure_file(FULL_PREFLIGHT)),
            "final_release_sha256": bytes_sha256(authority.release_raw),
            "execution_anchor_sha256": bytes_sha256(authority.anchor_raw),
            "consumed_capability_sha256": hash_identity(consumed_capability),
            "preregistration_sha256": hash_identity(secure_file(PREREG)),
            "runner_sha256": hash_identity(secure_file(Path(__file__))),
            "static_verification_sha256": hash_identity(secure_file(STATIC)),
            "v1_science_runner_sha256": V1_REVIEWED_SHA256["science_runner"],
            "v1_hold_receipt_sha256": V1_REVIEWED_SHA256["hold_receipt"],
            "bank_generation_receipt_sha256": bank_authority.receipt_sha256,
            "ordered_688_record_sha256": bank_authority.ordered_record_sha256,
            "target_bundle_sha256": DYNAMIC_HASHES["target_bundle_sha256"],
            "outer_fold_assignment_uint8_sha256": DYNAMIC_HASHES[
                "outer_fold_assignment_uint8_sha256"
            ],
            "selection_receipt_sha256": selection_hashes,
            "models_sha256": hash_identity(secure_file(models_path)),
        },
        "accuracy_first_amendment": {
            "all_H_entered_each_inner_selection": list(HS),
            "novelty_rank_energy_used_for_selection": False,
            "resource_residual_used_for_selection_or_accuracy_verdict": False,
        },
        "aggregate": final_summary,
        "fold_results": outer_receipts,
        "fold_gates": fold_gates,
        "gates": gates,
        "resource_observations_deployment_only": bank_authority.resource_observations,
        "novelty_observations_diagnostic_only": bank_authority.novelty_diagnostics,
        "bank_integrity_recomputed_before_target_decode": bank_authority.integrity_receipt,
        "native_storage_bytes_verified_after_claim": native_storage,
        "per_network_selected_H": selected_rows,
        "decision": (
            "GO_SIGNED_PAIR_ACCURACY_INTEGRATION_AND_RESOURCE_OPTIMIZATION_CHILD__NO_SUBMISSION"
            if passed
            else "CLOSE_THIS_SIGNED_PAIR_THREE_BANK_SPELLING__OPEN_SUPPORT_GE3_K6_CHILD__NO_RETRY"
        ),
    }
    result_path = temporary / "RESULT.json"
    write_exclusive_durable(result_path, canonical_json_bytes(result))
    directory = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    temporary.replace(BUNDLE)
    parent = os.open(HERE, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    ACTIVE_TEMP_BUNDLE = None
    print(
        json.dumps(
            {
                "status": result["status"],
                "pair_on_raw_mse": final_summary["pair_on_raw_mse"],
                "pair_zero_raw_mse": final_summary["pair_zero_raw_mse"],
                "fold_J_raw_mse": final_summary["fold_J_raw_mse"],
                "selected_H_by_fold": [row["selected_H"] for row in outer_receipts],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
