import ecole

class ObservationFunction(ecole.observation.NodeBipartite):

    def __init__(self, problem = None):
        super().__init__()

    def seed(self, seed):
        pass


# class Policy():

#     def __init__(self, problem = None):
#         self.rng = np.random.RandomState()

#         # get parameters
#         # params_path = f'agents/trained_models/{problem}/best_params.pkl'

#         # set up policy
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.policy = NeuralDiving().to(self.device)
#         #self.policy.load_state_dict(torch.load(params_path))

#     def seed(self, seed):
#         self.rng = np.random.RandomState(seed)

#     def __call__(self, action_set, observation):
#         # mask variable features (no incumbent info)
#         variable_features = observation.variable_features
#         variable_features = np.delete(variable_features, 14, axis=1)
#         variable_features = np.delete(variable_features, 13, axis=1)

#         constraint_features = torch.FloatTensor(observation.row_features).to(self.device)
#         edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64)).to(self.device)
#         edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1)).to(self.device)
#         variable_features = torch.FloatTensor(variable_features).to(self.device)
#         action_set = torch.LongTensor(np.array(action_set, dtype=np.int64)).to(self.device)

#         output, select = self.policy(constraint_features, edge_index, edge_attr, variable_features)
#         output = output[action_set]
#         select = select[action_set]

#         m = Binomial(1, select)
        
#         probs = m.sample()*output
#         m = Binomial(1, output[probs > 0])

#         action = (action_set[probs > 0], m.sample())
#         # action_idx = logits.argmax().item()
#         # action = action_set[action_idx]

#         return action