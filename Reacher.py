import os.path
import gurobipy as gp
from gurobipy import GRB
import scipy
import numpy
import gym
import mujoco_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import string
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # make render not lag
os.environ['TF_DETERMINISTIC_OPS'] = '1'
ewma_r = 0 # ewma for reward
arg_seed = 0
env_name = 'Reacher-v2'
env = gym.make(env_name)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.low
ewma_r = 0
arg_seed = 0
######################### seed ##############################
tf.compat.v1.reset_default_graph()
##################### hyper parameters ####################
meta_LR_C = 0.0001 # meta learning rate for critic
meta_LR_A = 0.001 # meta learning rate for actor
LR_C = 0.001  # learning rate for critic
LR_A = 0.0001 # learning rate for actor
GAMMA = 0.99 # determine the importance of future reward
TAU = 0.001

MEMORY_CAPACITY = 10000 # the size the replay buffer
BATCH_SIZE = 16
eval_freq = 5000
store_testing_before_action = []
store_testing_after_action = []

#################### optlayer #################################
def Projection(action, c):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as reacher_m:
            neta1 = action[0]
            neta2 = action[1]

            a1 = reacher_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
            a2 = reacher_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
            obj = (a1-neta1)**2 + (a2-neta2)**2
            reacher_m.setObjective(obj, GRB.MINIMIZE)

            # reacher_m.addConstr(a1+a2 <= 0.1)
            # reacher_m.addConstr(-0.1 <= a1+a2)
            reacher_m.addConstr((a1**2/(c[0]**2)+a2**2/(c[1]**2)) <= 1)

            reacher_m.optimize()

            return a1.X, a2.X

#################### testing part #################################
def evaluation(env_name, seed, ddpg, c, eval_episode=10):
    avgreward = 0
    avg = []
    eval_env = gym.make(env_name)
    eval_env.seed(seed+100)
    for eptest in range(eval_episode):
        running_reward = 0
        done = False
        s = eval_env.reset()
        while not done:
            action = ddpg.choose_action(s)
            store_testing_before_action.append(action)
            action, loss = Projection(action, c)
            store_testing_after_action.append(action)
            s_, r, done, info = eval_env.step(action)
            s = s_
            running_reward = running_reward+r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avgreward = avgreward+running_reward
        avg.append(running_reward)
    avgreward = avgreward/eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :", avgreward)
    print("------------------------------------------------")

    return avgreward

###############################  MAML  ####################################
class MAML(object):
    def __init__(self):
        
        configuration = tf.compat.v1.ConfigProto() # get configuration
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration) # use the configuration to approve session
        tf.random.set_seed(arg_seed)
        self.error_list = []

        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
        with tf.compat.v1.variable_scope('Critic'):
            self.q = self._build_c(
                self.S, self.a, scope='eval', trainable=True)

        # networks parameters
        self.a_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor')
        self.c_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic')

        # update actor(divided by |B|)
        self.actor_error = tf.reduce_sum(self.aerror_list)
        self.atrain = tf.compat.v1.train.AdamOptimizer(
            meta_LR_A).minimize(self.actor_error, var_list=self.a_params) # add optimizier

        # update critic
        # use meta_LR_C
        self.critic_error = tf.reduce_sum(self.cerror_list)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(
            meta_LR_C).minimize(self.critic_error, var_list=self.c_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def aerror_append(self, error):
        # for actor network
        self.error_list.append(error)
        
    def aerror_clean(self):
        # for actor network
        self.aerror_list = []

    def cerror_append(self, cerror):
        # for critic error
        self.cerror_list.append(cerror)
    
    def cerror_clean(self):
        # for critic network
        self.cerror_list = []
        
    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)  # a hidden layers with dim 400
            net2 = tf.compat.v1.layers.dense(
                net, 300, activation=tf.nn.relu, name='l2', trainable=trainable)    # a hidden layers with dim 300
            a = tf.compat.v1.layers.dense(
                net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable) # the last layer that output result
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 400, activation=tf.nn.relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable(
                'w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable(
                'w2_a', [self.a_dim, 300], trainable=trainable)
            b2 = tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2 = tf.nn.relu(tf.matmul(a, w2_a)+tf.matmul(net, w2_net)+b2)
            # Q(s,a)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)

    # do the gradient
    def gradient(self):
        self.sess.run(self.atrain)
        self.sess.run(self.ctrain)

###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, c):
        
        # initial envirnment here
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1+1), dtype=np.float32)
        self.pointer = 0
        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        tf.random.set_seed(arg_seed)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.c = c
        
        self.S = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], 's')   # current state
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], 's_') # next state
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')   # reward
        self.Done = tf.compat.v1.placeholder(tf.float32, [None, 1], 'done') # if the task is done

        # create the unique name in order not to get the same scope
        self.length_of_name = 5
        self.actor_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(self.length_of_name))
        self.critic_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(self.length_of_name))

        with tf.compat.v1.variable_scope(self.actor_name):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope(self.critic_name):
            self.q = self._build_c(
                self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.actor_name+"/eval")
        self.at_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.actor_name+"/target")
        self.ce_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.critic_name+"/eval")
        self.ct_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.critic_name+"/target")

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + (1-self.Done)*GAMMA * q_
        td_error = tf.compat.v1.losses.mean_squared_error(
            labels=q_target, predictions=self.q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(
            LR_C).minimize(td_error, var_list=self.ce_params) # minimize the loss

        a_loss = - tf.reduce_mean(input_tensor=self.q) # maximize the q

        self.atrain = tf.compat.v1.train.AdamOptimizer(
            LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):

        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        buffer_size = min(self.pointer+1, MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        bs_ = bt[:, self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+1+self.s_dim]
        bd = bt[:, -1:]
        self.sess.run(self.atrain, {self.S: bs})
        return self.sess.run(
            self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.Done: bd})

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 400, activation=tf.nn.relu, name='l1', trainable=trainable) # a hidden layers with dim 400
            net2 = tf.compat.v1.layers.dense(
                net, 300, activation=tf.nn.relu, name='l2', trainable=trainable) # a hidden layers with dim 300
            a = tf.compat.v1.layers.dense(
                net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable) # the last layer that output result
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(
                s, 400, activation=tf.nn.relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable(
                'w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable(
                'w2_a', [self.a_dim, 300], trainable=trainable)
            b2 = tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2 = tf.nn.relu(tf.matmul(a, w2_a)+tf.matmul(net, w2_net)+b2)
            # Q(s,a)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)
    
    def get_gradient(self, s, a):
        return self.sess.run(self.action_grad, {self.S: s, self.a: a})

    def fw_update(self):
        lr = 0.05
        buffer_size = min(ddpg.pointer+1, MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = self.sess.run(self.a, {self.S: bs})
        #######let ba in constraint######
        for i in range(BATCH_SIZE):
            ba[i] = Projection(ba[i], self.c) # need to add the parameter

        #######let ba in constraint######
        grad = self.get_gradient(bs, ba)
        grad = np.squeeze(grad)
        action_table = np.zeros((BATCH_SIZE, 2))  # (16,16,2
        for i in range(BATCH_SIZE):
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model(env=env) as fw_m:
                    a1 = fw_m.addVar(lb=-1, ub=1, name="a1",
                                     vtype=GRB.CONTINUOUS)
                    a2 = fw_m.addVar(lb=-1, ub=1, name="a2",
                                     vtype=GRB.CONTINUOUS)
                    obj = a1*grad[i][0]+a2*grad[i][1]
                    fw_m.setObjective(obj, GRB.MAXIMIZE)
                    # fw_m.addConstr(a1+a2 <= 0.1)
                    # fw_m.addConstr(-0.1 <= a1+a2)
                    fw_m.addConstr((a1**2+a2**2) <= 0.05)
                    fw_m.optimize()
                    action_table[i][0] = a1.X
                    action_table[i][1] = a2.X
                action_table[i] = action_table[i]*lr+ba[i] * (1-lr)
        self.sess.run(self.update, {self.S: bs, self.tf_table: action_table})

# contraints
constr_pool = []  # create a ddpg_pool of tasks
len_pool = 10
for i in range(len_pool):
    c = []
    c.append(1 - random.uniform(0, 1)) # to exclude 0
    c.append(1 - random.uniform(0, 1))
    constr_pool.append(c)

env_name = 'Reacher-v2'

ewma = []
store_before_action = []
store_before_action_and_gaussain = []

store_after_action = []
eva_reward = []
reward = []
Net_action = np.zeros((100000, a_dim+2))
max_action = float(env.action_space.high[0])

step = 0
constr_ep = 100

for _ in range(constr_ep):
    constr_batchSize = 3 # define num of tasks in a batch
    constr_batch = random.sample(constr_pool, constr_batchSize)
    for c in constr_batch:
        maml = MAML()
        for ep in range(10000000):
            # env.render()
            ddpg = DDPG(a_dim, s_dim, a_bound, c)

            R = 0
            done = False
            s = env.reset()
            step = 0
            while not done:
                step = step+1
                if ddpg.pointer < 1000:
                    action = env.action_space.sample()
                else:
                    action = ddpg.choose_action(s)
                    store_before_action_and_gaussain.append(action)
                    action = (action+np.random.normal(0, 0.1, a_dim)
                            ).clip(-max_action, max_action)

                store_before_action.append(action)
                action = Projection(action, c)
                store_after_action.append(action)

                s_, r, done, info = env.step(action)

                done_bool = False if step == env._max_episode_steps else done

                ddpg.store_transition(s, action, r, s_, done_bool)
                if ddpg.pointer >= 1000:
                    learn_error = ddpg.learn() # critic error
                    fw_error = ddpg.fw_update() # actor error

                    # append the actor error (loss)
                    maml.aerror_append(fw_error)
                    # append the critic error
                    maml.cerror_append(learn_error)

                if (ddpg.pointer+1) % eval_freq == 0:
                    eva_reward.append(evaluation(env_name, arg_seed, ddpg))
                s = s_
                R += r

            ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
            print({
                'episode': ep,
                'reward': R,
                'ewma_reward': ewma_r
            })
            reward.append(R)
            ewma.append(ewma_r)
            if(ddpg.pointer >= 500000):
                print("done training")
                break
        maml.error_clean()

np.save("Reacher_{}_DDPGFw_Reward".format(arg_seed), reward)
np.save("Reacher_{}_DDPGFw_before_Action".format(
    arg_seed), store_before_action)
np.save("Reacher_{}_DDPGFw_after_Action".format(arg_seed), store_after_action)
np.save("Reacher_{}_DDPGFw_eval_reward".format(arg_seed), eva_reward)
np.save("Reacher_{}_DDPGFw_before_Action_Gaussian".format(
    arg_seed), store_before_action_and_gaussain)
np.save("Reacher_{}_DDPGFW_eval_before_Action".format(
    arg_seed), store_testing_before_action)
np.save("Reacher_{}_DDPGFW_eval_After_Action".format(
    arg_seed), store_testing_after_action)