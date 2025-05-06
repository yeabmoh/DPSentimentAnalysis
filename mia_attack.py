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

# ✅ Evaluate attack performance on shadow models
y_attack_pred = attack_model.predict(X_attack)
print("Attack Accuracy:", accuracy_score(y_attack, y_attack_pred))
print("Attack Precision:", precision_score(y_attack, y_attack_pred, zero_division=0))
print("Attack Recall:", recall_score(y_attack, y_attack_pred, zero_division=0))

# ✅ Evaluate on target model
target_probs = np.load('target_model_probs.npy')
target_membership_labels = np.load('target_model_labels.npy')
y_target_pred = attack_model.predict(target_probs)

print("Target Attack Accuracy:", accuracy_score(target_membership_labels, y_target_pred))
print("Target Attack Precision:", precision_score(target_membership_labels, y_target_pred, zero_division=0))
print("Target Attack Recall:", recall_score(target_membership_labels, y_target_pred, zero_division=0))
