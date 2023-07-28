## Progress

<ol>
  <li>For the first task, I had to study and run the Monte Carlo Tree Search code for self assembling peptides, which I did.<a href = 'https://drive.google.com/file/d/1kmlz5kaJnC4FblE5oBK-0NnJRRSLm_pI/view?usp=drivesdk'> Paper</a>, <a href = 'https://zenodo.org/record/6564202'>Code </a></li>
  <li>Next, I started studying about Reinforcement Learning, to get a brief detail of the difference b/w RL and Supervised Learning by following <a href = "https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ">Google DeepMind's YouTube playlist</a>. In this playlist, I got the basic idea behind policies, actions, rewards, environment and observation spaces.</li>
  <li>After that, I started to learn about the RL's Policy Gradient Method, following the same playlist, getting an idea about the gradient ascent method.</li>
  <li>Next, I went to understand how to code the method, using observation spaces, actions, and rewards, by following the article coding the solution to OpenAI Gymnasium's <a href= "https://gymnasium.farama.org/environments/classic_control/cart_pole/">CartPole Problem</a>. <a href= "https://towardsdatascience.com/policy-gradient-methods-104c783251e0">Article</a></li>
  <li>After understanding the code, I have started to implement it similarly to the self assembling peptide problem. I have changed the environment as it was pre defined in the CartPole problem but I had to think how to define it in the peptide problem.</li> 
  <li>I have made the <a href="https://github.com/ajiteshshree/summer_internship_2023/blob/main/rlPeptide/model.ipynb">first draft</a> of my code and have to add the reward policy, specifically following the Aggregate Propensity and logP values as they were the policies used in the MCTS paper.</li> 
  <li>Also, as Professor Bapi advised, I think of changing the no.of hidden layers of my code ( currently 1) and the no. of nodes in it (currently 128) after I figure out adding the reward policies.</li>
</ol>
