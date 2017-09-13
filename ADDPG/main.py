from model import *
import sys
import os
import multiprocessing
import threading
import argparse
worker_threads = []

def main():

    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
    parser.add_argument('--num_workers', dest='num_workers',action='store',default=3,type=int)
    parser.add_argument('--visualize', dest='vis', action='store_true', default=False)
    args = parser.parse_args()

    load_model = args.load_model
    num_workers = args.num_workers
    vis = args.vis
    training = True#not load_model
    model_path = './models'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
	
    # hyperparameters
    explore = 1000
    batch_size = 32
    gamma = 0.995
    n_step = 3
        
    tf.reset_default_graph()
        
    with tf.Session() as sess:
	    with tf.device("/cpu:0"): 
           	global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
	        global_actor_network = ActorNetwork(sess,41+14+3,18,'global'+'/actor')
	        num_cpu = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
	        workers = []
	        # Create worker classes
	        for i in range(num_workers):
	            worker = Worker(sess,i,model_path,global_episodes,explore,training,vis,batch_size,gamma,n_step,global_actor_network.net)
		    workers.append(worker)
	        saver = tf.train.Saver()

            coord = tf.train.Coordinator()
            if load_model == True:
                print ('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
            	saver.restore(sess,ckpt.model_checkpoint_path)
            	print ('Loading Model succeeded...')
            else:
                sess.run(tf.global_variables_initializer())
                
            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate thread.
            
            for worker in workers:
                worker_work = lambda: worker.work(coord,saver)
                t = threading.Thread(target=(worker_work))
                t.daemon = True
                t.start()
                worker_threads.append(t)
                sleep(0.05)
            coord.join(worker_threads)
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-c received! Sending kill to threads...")
        for t in worker_threads:
            t.kill_received = True
