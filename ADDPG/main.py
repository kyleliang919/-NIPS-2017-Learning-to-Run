

from model import *
import sys
import os
import multiprocessing
import threading

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():

    if len(sys.argv)==1:
	num_workers = 1
    else:
	num_workers = int(sys.argv[1])

    load_model = False
    training = not load_model
    model_path = './models'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
	
    explore = 1000
    decay = 1.

    if load_model:
	agent.load_model(model_path)
	decay -= 45*1./EXPLORE # change Aug20
        
    tf.reset_default_graph()
        
    with tf.Session() as sess:
	with tf.device("/cpu:0"): 
       	    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
	    global_actor_network = ActorNetwork(sess,41,18,'global'+'/actor')
	    global_critic_network = CriticNetwork(sess,41,18,'global'+'/critic')
	    num_cpu = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
	    workers = []
	    # Create worker classes
	    for i in range(num_workers):
	        worker = Worker(sess,i,model_path,global_episodes,explore,decay,training)
		workers.append(worker)
		worker.start(setting=0)
	    saver = tf.train.Saver(max_to_keep=5)

        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)
        
if __name__ == "__main__":
    main()
