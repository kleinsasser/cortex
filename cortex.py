import numpy as np

class LearningObject:
    def __init__(self, value):
        self.learning = value
    
    def set_learning(self, value):
        self.learning = value

class VisualActivationThreshold:
    def __init__(self, threshold, decrement):
        self.starting_threshold = threshold
        self.threshold = threshold
        self.decrement = decrement

    def decrement_threshold(self):
        self.threshold = self.threshold - self.decrement
    
    def reset(self):
        self.threshold = self.starting_threshold

'''
Columns are the building blocks of the Cortex. They are meant to recognize and "activate" in response
to specific sensory inputs, and also form weighted connections with other columns in the cortex.
'''
class Column:
    def __init__(self, learning_object):
        self.input_type = None
        self.reference = None

        self.connections = []

        self.learning = learning_object

        self.activation = 0
        self.propagated = False

    def propagate_activation(self):
        for k, weight in self.connections:
            k.activation += self.activation * weight
        self.propagated = True

    def reset(self):
        self.activation = 0
        self.propagated = False

'''
A Column for recognizing specific textual input. Activation is determined based on whether an input sample
is equal to the column's reference.
'''
class TextColumn(Column):
    def __init__(self, learning_object, reference):
        super().__init__(learning_object)
        self.input_type = 'text'
        self.reference = str(reference)
    
    def activate(self, sample):
        # get activation for self
        if sample != self.reference:
            self.activation = 0.0
        else:
            self.activation = 1.0

    def set_reference(self, ref):
        self.reference = str(ref)

'''
Column designed to take visual input, specifically 20x20 images, but really it accepts any nd.array. Inputs should
be normalized to 0.0-1.0. Activation is determined by Mean Absolute Error and an activation threshold.
'''
class VisionColumn(Column):
    def __init__(self, learning_object, min_activation, reference):
        super().__init__(learning_object)
        self.learning = learning_object

        self.support = 1
        self.input_type = 'visual'
        self.min_activation = min_activation
        self.reference = reference

    '''
    Determines whether a visual input should activate the column (whether the column "recognizes" the input). If so,
    also incorporates the input into its reference ("learns" the sample).
    '''
    def activate(self, sample):
        self.activation = 1 - np.mean(np.abs(self.reference - sample))

        if self.activation < self.min_activation.threshold:
            self.activation = 0
        elif self.learning.learning:
            self.learn_sample(sample)

    '''
    Incorporates a given input sample to the column's reference through a weighted average.
    '''
    def learn_sample(self, sample):
        if self.support < 1:
            self.reference = sample
        else:
            self.reference = np.average((self.reference, sample), weights=(self.support, 1), axis=0)
        self.support += 1

'''
UNIMPLEMENTED: Meant to take the activations of other columns as input, activating only when a combination of inputs
occurs.
'''
class CombinationColumn(Column):
    def __init__(self, learning_object):
        super().__init__(learning_object)

    def activate(self, sample):
        pass

    def learn_sample(self, sample):
        pass

'''
Cortex class. Implements the functionality for intelligiently making associations between Column objects.
'''
class Cortex:
    def __init__(self, min_visual_activation = 0.7, visual_activation_decrement = 0.05, min_text_activation = 0.9, default_edge_weight = 1.0, boost_constant = 1.1, weaken_constant = 0.91):
        self.columns = []
        self.learning = LearningObject(True)

        self.min_visual_activation = VisualActivationThreshold(min_visual_activation, visual_activation_decrement)
        self.min_text_activation = min_text_activation

        self.default_edge_weight = default_edge_weight

        self.boost_constant = boost_constant
        self.weaken_constant = weaken_constant

    '''
    The primary method used by the Cortex to learn. Takes 2 sensory inputs, feeds them individually to the Cortex.
    If either input is not recognized by the Cortex, a new column referencing the sensory input is added to the Cortex.
    The columns activated by each sensory input individually are stored, and connections are created or strengthened
    between them.
    '''
    def learn_association(self, input_tuple1, input_tuple2):
        input_type1, input1 = input_tuple1
        input_type2, input2 = input_tuple2

        self.percieve_input(input_type = input_type1, input = input1)
        perception1 = self.get_active_cols()
        if len(perception1) < 1:
            perception1 = [self.add_column(input_type = input_type1, input = input1)]
        self.reset()

        self.percieve_input(input_type = input_type2, input = input2)
        perception2 = self.get_active_cols()
        if len(perception2) < 1:
            perception2 = [self.add_column(input_type = input_type2, input = input2)]
        self.reset()
        
        for i in perception1:
            for j in perception2:
                self.boost_connection(i, j)

    '''
    Sends a sensory input to the Cortex, activating the columns whose references match said input.
    '''
    def percieve_input(self, input_type, input):
        for c in self.columns:
            if c.input_type != input_type: continue
            c.activate(input)
    
    '''
    Parses the Corex's columns and returns a list of those columns whose activation is not 0, optionally filtering by 
    input_type.
    '''
    def get_active_cols(self, input_type = None):
        cols = []
        for c in self.columns:
            if input_type != None and c.input_type != input_type: continue
            if c.activation > 0: cols.append(c)
        return cols

    '''
    Returns the most active column in the Cortex subject to an optional input_type filter. 
    '''
    def get_most_active_col(self, input_type = None):
        cols = self.get_active_cols(input_type)

        if len(cols) < 1: return None
        return max(cols, key = lambda x: x.activation)

    '''
    Parses the Cortex and propogates the activations of active columns to the columns with which they connect.
    '''
    def propagate_activations(self):
        set_cols = []
        for c in self.columns:
            if c.activation > 0 and not c.propagated:
                set_cols.append(c)
        
        for c in set_cols:
            c.propagate_activation()

    '''
    Adds a column to the Cortex.
    '''
    def add_column(self, input_type, input):
        c = None
        if input_type == 'visual':
            c = VisionColumn(self.learning, self.min_visual_activation, input)
        
        if input_type == 'text':
            c = TextColumn(self.learning, input)
        
        self.columns.append(c)
        return c

    '''
    Boosts the connection of two columns in the Cortex, creating a default-strength connection if one
    doesn't already exist.
    '''
    def boost_connection(self, column1, column2):
        exists = False
        for k in column1.connections:
            if k[0] == column2: 
                new_connection = (k[0], k[1] + self.boost_constant)
                column1.connections.remove(k)
                column1.connections.append(new_connection)
                exists = True
                break
        
        for k in column2.connections:
            if k[0] == column1: 
                new_connection = (k[0], k[1] + self.boost_constant)
                column2.connections.remove(k)
                column2.connections.append(new_connection)
                break
        
        if not exists:
            column1.connections.append((column2, self.default_edge_weight))
            column2.connections.append((column1, self.default_edge_weight))

    '''
    Sets the activation of each column in the network to 0, and sets the propagated property of each column
    to False.
    '''
    def reset(self):
        for c in self.columns:
            c.reset()

    '''
    Function for forcing the Cortex to recognize a visual input by iteratively lowering the threshold for
    recognition in visual columns.
    '''
    def force_visual_perception(self, image):
        cols = []
        first = True
        while len(cols) < 1:
            if first:
                first = False
            else:
                self.min_visual_activation.decrement_threshold()
            self.percieve_input(input_type = 'visual', input = image)
            cols = self.get_active_cols()
        
        self.min_visual_activation.reset()
        return cols