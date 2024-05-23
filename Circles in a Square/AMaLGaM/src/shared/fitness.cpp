//
// Created by Renzo on 12-6-2020.
//

/**
 * Returns 1 if x is better than y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x has a smaller objective value than y
 */
short betterFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y )
{
  short result;
  
  result = 0;
  
  if( constraint_value_x > 0 ) /* x is infeasible */
  {
    if( constraint_value_y > 0 ) /* Both are infeasible */
    {
      if( constraint_value_x > constraint_value_y )
        result = 1;
    }
  }
  else /* x is feasible */
  {
    if( constraint_value_y > 0 ) /* x is feasible and y is not */
      result = 1;
    else /* Both are feasible */
    {
      if( objective_value_x > objective_value_y )
        result = 1;
    }
  }
  
  return( result );
}
