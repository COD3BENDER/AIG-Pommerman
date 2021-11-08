Group AK - Modified MCTS
This agent is modified by the implementation of Progressive Bias, Decaying Rewards and Action selection changes
To initialize this agent add the line of code below to the switch statement in the run file and also import the class

	import players.groupAK.MCTSParamsTD;
	import players.groupAK.MCTSPlayerTD;

         case 7:
		MCTSParamsTD mctsParamsTD = new MCTSParamsTD();
		mctsParamsTD.stop_type = mctsParamsTD.STOP_ITERATIONS;
		mctsParamsTD.num_iterations = 200;
		mctsParamsTD.rollout_depth = 12;

		mctsParamsTD.heuristic_method = mctsParamsTD.CUSTOM_HEURISTIC;
		p = new MCTSPlayerTD(seed, playerID++, mctsParamsTD);
		playerStr[i-4] = "G-AK";
		break;

To initialize this agent to the test folder add the line of code below to the main method and import class
	
	import players.groupAK.MCTSParamsTD;
	import players.groupAK.MCTSPlayerTD;

	 MCTSParamsTD mctsParamsTD = new MCTSParamsTD();
	 mctsParamsTD.stop_type = mctsParamsTD.STOP_ITERATIONS;
 	 mctsParamsTD.heuristic_method = mctsParamsTD.CUSTOM_HEURISTIC;

	players.add(new MCTSPlayerTD(seed, playerID++, new MCTSParamsTD()));// to add player into the game

This Package is able to switch between the three action selection methods for rollout you can change it by setting the 
variable rolloutType in the singleTreeNode class to either 0,1 or 2. you can also change the value of gamma in that class too.
choose 0 for default action selection 1 for OSLA action selection and 2 for Modified Action selection used in testing

