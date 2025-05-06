import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load shadow model outputs
shadow_train_probs = np.load('shadow_model_outputs/shadow_model_0_train_probs.npy')
shadow_nontrain_probs = np.load('shadow_model_outputs/shadow_model_0_nontrain_probs.npy')
shadow_train_labels = np.load('shadow_model_outputs/shadow_model_0_train_labels.npy')
shadow_nontrain_labels = np.load('shadow_model_outputs/shadow_model_0_nontrain_labels.npy')


# Combine into attack dataset
X_attack = np.vstack([shadow_train_probs, shadow_nontrain_probs])
y_attack = np.concatenate([shadow_train_labels, shadow_nontrain_labels])

# Train attack model
attack_model = LogisticRegression()
attack_model.fit(X_attack, y_attack)

# Evaluate attack performance on shadow models
y_attack_pred = attack_model.predict(X_attack)
print("Attack Accuracy:", accuracy_score(y_attack, y_attack_pred))
print("Attack Precision:", precision_score(y_attack, y_attack_pred))
print("Attack Recall:", recall_score(y_attack, y_attack_pred))

# Evaluate on target model
target_probs = np.load('target_model_probs.npy')
target_membership_labels = np.load('target_model_labels.npy')
y_target_pred = attack_model.predict(target_probs)

print("Target Attack Accuracy:", accuracy_score(target_membership_labels, y_target_pred))
print("Target Attack Precision:", precision_score(target_membership_labels, y_target_pred))
print("Target Attack Recall:", recall_score(target_membership_labels, y_target_pred))