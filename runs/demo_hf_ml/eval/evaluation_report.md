# Evaluation report

## 1. Inter-agent consistency
| Metric | Value |
|---|---|
| ta_kappa_A_B | 0.837 |
| ta_raw_agreement | 0.923 |
| ta_arbitrated_share | 0.184 |
| ta_screener_A_include_rate | 0.401 |
| ta_screener_B_include_rate | 0.368 |
| evidence_quotes_verified_in_source | 0.842 |
| critic_revision_rounds | 7 |
| invalid_citations_final | 0 |
| uncited_included_studies_final | 0 |

## 2. Completeness
| Metric | Value |
|---|---|
| pipeline_stages_completed | 8 |
| pipeline_stages_total | 8 |
| fulltext_retrieval_rate | 0.398 |
| studies_extracted_vs_included | 10/71 |
| included_assessed_on_full_text | 7 |
| included_assessed_on_abstract_only | 3 |
| extraction_field_fill_rate_mean | 0.679 |
| prisma_items_present | 16/16 |
| prisma_items_missing | [] |
| decisions_with_rationale | 1.0 |

**Extraction fill rate by field**

| Field | Fill rate |
|---|---|
| country | 0.5 |
| data_source | 0.9 |
| sample_size | 0.8 |
| hf_population | 0.8 |
| outcome | 1.0 |
| ml_models | 1.0 |
| best_model | 1.0 |
| best_auc | 0.8 |
| best_auc_ci | 0.2 |
| comparator | 0.8 |
| comparator_auc | 0.5 |
| validation | 0.7 |
| calibration_reported | 0.1 |
| explainability | 0.4 |

## 3. Screening accuracy vs blinded human reviewer
| Metric | Value |
|---|---|
| n | 30 |
| tp | 11 |
| tn | 6 |
| fp | 4 |
| fn | 9 |
| accuracy | 0.567 |
| sensitivity_recall | 0.55 |
| specificity | 0.6 |
| precision | 0.733 |
| kappa_ai_vs_human | 0.133 |

## 4. Extraction accuracy vs human verification
| Metric | Value |
|---|---|
| fields_checked | 1 |
| correct | 1 |
| incorrect | 0 |
| unverifiable | 0 |
| accuracy | 1.0 |
| by_field | {'sample_size': '1/1'} |

## 5. Time
| Metric | Value |
|---|---|
| first_event | 2026-11-12 23:30:19+00:00 |
| last_event | 2026-11-13 13:01:00+00:00 |
| elapsed_hours_wall_clock | 13.51 |
| active_hours_pipeline | 2.07 |
| human_interventions | 10 |
| human_overrides | 0 |
| model_switches | 2 |
| assumptions_manual_minutes | {'per_abstract_per_reviewer': 1.0, 'per_fulltext_per_reviewer': 10.0, 'per_study_extraction': 30.0, 'dual_review': True} |
| manual_estimate_hours_screening_and_extraction | 43.4 |
| reference_benchmark | Borah et al., BMJ Open 2017: mean 67.3 weeks to complete and publish a systematic review (195 PROSPERO reviews). |
| speedup_vs_manual_estimate | 21.0 |
