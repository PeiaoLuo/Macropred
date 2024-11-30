If you want to develop new model,
there are three part need to update:
like if your model name is XXX

1. go to data_prep/model_prep.py to update your data preparation method for this model,
   add a new function in the class
   the name should be strictly align with structure:
   ```python
   def XXX(self, y_train, X_train, X_test, other with default val args):
   	#your process
   	return y_train, X_train, X_test
   ```
2. go to models/shallow (or deep).py to add your new model into the class,
   shoud strictly align with the structure:
   ```python
   def XXX(self, X_train, X_test, y_train, true_y, targetname, plot: bool=True) -> None:
   	from general_modules.models.shallow_eval import XXX_eval
   	#your model realization
   	#your storage of results
   	return None
   ```
3. go to models/shallow_eval (or deep_eval).py to add your evaluation, this is according to how you will use it in step 2, any is ok
