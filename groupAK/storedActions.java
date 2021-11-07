package players.groupAK;
// i tried to improve the action selection further by storing
// previous integers of actions and its evaluation into a list and then returning the action with the highest evaluation
// but since hashmaps only store a key and value it was much harder to implement there were libraries online such as multimap
// but wanted to improve the agent without the use of external libraries.

// the idea was to improve the action selection by storing all previous actions with its evaluation and returning the highest evaluation.
// Then comparing the new random action with the highest previous action. for example storing the last 5 - 10 actions chosen then comparing a
// new random action to the highest of the 5 with the use of a boost as the storage of number gets closer to 5 - 10 as they would be more relevant
public class storedActions {
    public int actionP;
    public double EvalP;
    public storedActions(int action, double evaluation){ // a wrapper class created to store 2 values together in a list as java doesnt have tuples like python numpy
        this.actionP = action;
        this.EvalP = evaluation;

    }
}
