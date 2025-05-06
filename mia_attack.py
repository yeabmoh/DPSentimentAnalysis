import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ✅ Load all shadow model outputs and combine
shadow_train_probs_all = np.vstack([
    np.load('shadow_model_outputs/shadow_model_0_train_probs.npy'),
    np.load('shadow_model_outputs/shadow_model_1_train_probs.npy'),
    np.load('shadow_model_outputs/shadow_model_2_train_probs.npy')
])
shadow_nontrain_probs_all = np.vstack([
    np.load('shadow_model_outputs/shadow_model_0_nontrain_probs.npy'),
    np.load('shadow_model_outputs/shadow_model_1_nontrain_probs.npy'),
    np.load('shadow_model_outputs/shadow_model_2_nontrain_probs.npy')
])
shadow_train_labels_all = np.concatenate([
    np.load('shadow_model_outputs/shadow_model_0_train_labels.npy'),
    np.load('shadow_model_outputs/shadow_model_1_train_labels.npy'),
    np.load('shadow_model_outputs/shadow_model_2_train_labels.npy')
])
shadow_nontrain_labels_all = np.concatenate([
    np.load('shadow_model_outputs/shadow_model_0_nontrain_labels.npy'),
    np.load('shadow_model_outputs/shadow_model_1_nontrain_labels.npy'),
    np.load('shadow_model_outputs/shadow_model_2_nontrain_labels.npy')
])

# Combine into attack dataset
X_attack = np.vstack([shadow_train_probs_all, shadow_nontrain_probs_all])
y_attack = np.concatenate([shadow_train_labels_all, shadow_nontrain_labels_all])

# ✅ Train attack model
attack_model = LogisticRegression()
attack_model.fit(X_attack, y_attack)

# Evaluate on non-DP target model
target_non_dp_probs = np.load('target_non_dp_probs.npy')
target_non_dp_labels = np.load('target_non_dp_labels.npy')
y_non_dp_scores = attack_model.predict_proba(target_non_dp_probs)[:, 1]
y_non_dp_pred = (y_non_dp_scores > 0.5).astype(int)

print("\n[Non-DP Target Attack]")
print("Target Attack Accuracy:", accuracy_score(target_non_dp_labels, y_non_dp_pred))
print("Target Attack Precision:", precision_score(target_non_dp_labels, y_non_dp_pred, zero_division=0))
print("Target Attack Recall:", recall_score(target_non_dp_labels, y_non_dp_pred, zero_division=0))

# Evaluate on DP target model (with probability thresholding)
target_dp_probs = np.load('target_dp_probs.npy')
target_dp_labels = np.load('target_dp_labels.npy')
y_dp_scores = attack_model.predict_proba(target_dp_probs)[:, 1]
y_dp_pred = (y_dp_scores > 0.5).astype(int)

print("\n[DP Target Attack]")
print("Target Attack Accuracy:", accuracy_score(target_dp_labels, y_dp_pred))
print("Target Attack Precision:", precision_score(target_dp_labels, y_dp_pred, zero_division=0))
print("Target Attack Recall:", recall_score(target_dp_labels, y_dp_pred, zero_division=0))

print("DP predicted counts:", np.unique(y_dp_pred, return_counts=True))

print("non-DP predicted counts:", np.unique(y_non_dp_pred, return_counts=True))
