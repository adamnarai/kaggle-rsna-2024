import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics
import torch
import torch.nn.functional as F


class ParticipantVisibleError(Exception):
    pass


def get_condition(full_location: str) -> str:
    # Given an input like spinal_canal_stenosis_l1_l2 extracts 'spinal'
    for injury_condition in ['spinal', 'foraminal', 'subarticular']:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f'condition not found in {full_location}')


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        any_severe_scalar: float
    ) -> float:
    '''
    Pseudocode:
    1. Calculate the sample weighted log loss for each medical condition:
    2. Derive a new any_severe label.
    3. Calculate the sample weighted log loss for the new any_severe label.
    4. Return the average of all of the label group log losses as the final score, normalized for the number of columns in each group.
       This mitigates the impact of spinal stenosis having only half as many columns as the other two conditions.
    '''

    target_levels = ['normal_mild', 'moderate', 'severe']

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission[target_levels].values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission[target_levels].values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    solution['study_id'] = solution['row_id'].apply(lambda x: x.split('_')[0])
    solution['location'] = solution['row_id'].apply(lambda x: '_'.join(x.split('_')[1:]))
    solution['condition'] = solution['row_id'].apply(get_condition)

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)

    submission['study_id'] = solution['study_id']
    submission['location'] = solution['location']
    submission['condition'] = solution['condition']

    condition_losses = []
    condition_weights = []
    for condition in ['spinal', 'foraminal', 'subarticular']:
        condition_indices = solution.loc[solution['condition'] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, 'sample_weight'].values,
        )
        condition_losses.append(condition_loss)
        condition_weights.append(1)

    any_severe_spinal_labels = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['severe'].max())
    any_severe_spinal_weights = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['sample_weight'].max())
    any_severe_spinal_predictions = pd.Series(submission.loc[submission['condition'] == 'spinal'].groupby('study_id')['severe'].max())
    any_severe_spinal_loss = sklearn.metrics.log_loss(
        y_true=any_severe_spinal_labels,
        y_pred=any_severe_spinal_predictions,
        sample_weight=any_severe_spinal_weights
    )
    condition_losses.append(any_severe_spinal_loss)
    condition_weights.append(any_severe_scalar)
    return np.average(condition_losses, weights=condition_weights)


def prepare_data(y, pred):
    out_vars = [
        'spinal_canal_stenosis_l1_l2',
        'spinal_canal_stenosis_l2_l3',
        'spinal_canal_stenosis_l3_l4',
        'spinal_canal_stenosis_l4_l5',
        'spinal_canal_stenosis_l5_s1',
        'left_neural_foraminal_narrowing_l1_l2',
        'left_neural_foraminal_narrowing_l2_l3',
        'left_neural_foraminal_narrowing_l3_l4',
        'left_neural_foraminal_narrowing_l4_l5',
        'left_neural_foraminal_narrowing_l5_s1',
        'right_neural_foraminal_narrowing_l1_l2',
        'right_neural_foraminal_narrowing_l2_l3',
        'right_neural_foraminal_narrowing_l3_l4',
        'right_neural_foraminal_narrowing_l4_l5',
        'right_neural_foraminal_narrowing_l5_s1',
        'left_subarticular_stenosis_l1_l2',
        'left_subarticular_stenosis_l2_l3',
        'left_subarticular_stenosis_l3_l4',
        'left_subarticular_stenosis_l4_l5',
        'left_subarticular_stenosis_l5_s1',
        'right_subarticular_stenosis_l1_l2',
        'right_subarticular_stenosis_l2_l3',
        'right_subarticular_stenosis_l3_l4',
        'right_subarticular_stenosis_l4_l5',
        'right_subarticular_stenosis_l5_s1',
    ]

    y_one_hot = F.one_hot(y, 3).cpu().numpy().astype(np.float32)

    if pred.dim() == 2:
        pred = torch.unflatten(pred, 1, [3, -1])
    pred_reshaped = pred.swapaxes(1, 2).softmax(-1).cpu().numpy().astype(np.float32)
    solution_list = []
    submission_list = []
    for i in range(y_one_hot.shape[0]):
        solution_list.append(
            pd.DataFrame(
                {
                    'row_id': [f'{i}_{x}' for x in out_vars],
                    'normal_mild': y_one_hot[i, :, 0],
                    'moderate': y_one_hot[i, :, 1],
                    'severe': y_one_hot[i, :, 2],
                    'sample_weight': 2 ** y_one_hot[i].argmax(axis=1),
                }
            )
        )
        submission_list.append(
            pd.DataFrame(
                {
                    'row_id': [f'{i}_{x}' for x in out_vars],
                    'normal_mild': pred_reshaped[i, :, 0],
                    'moderate': pred_reshaped[i, :, 1],
                    'severe': pred_reshaped[i, :, 2],
                }
            )
        )
    solution = pd.concat(solution_list, ignore_index=True)
    submission = pd.concat(submission_list, ignore_index=True)

    return solution, submission
