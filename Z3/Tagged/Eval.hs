module Z3.Tagged.Eval (EvalAst, eval, evalT, mapEval,
                       evalBool, evalInt, evalReal, evalBv, evalFunc,
                       runWithModel) where

import Control.Monad.Trans.Maybe
import Control.Monad.Trans.Reader
import Z3.Tagged (Z3, Model, AST, FuncDecl, FuncModel)
import qualified Z3.Tagged as T

-------------------------------------------------
-- ** Helpers

-- | Type of an evaluation function for '(AST s)'.
--
-- Evaluation may fail (i.e. return 'Nothing') for a few
-- reasons, see 'modelEval'.
type EvalAst s a = AST s -> Eval s a
type Eval s = ReaderT (Model s) (MaybeT (Z3 s))

-- | An alias for 'modelEval' with model completion enabled.
eval :: EvalAst s (AST s)
eval = liftEval T.eval

-- | Evaluate an (AST s) node of sort /bool/ in the given model.
--
-- See 'modelEval' and 'getBool'.
evalBool :: EvalAst s Bool
evalBool = liftEval T.evalBool

-- | Evaluate an (AST s) node of sort /int/ in the given model.
--
-- See 'modelEval' and 'getInt'.
evalInt :: EvalAst s Integer
evalInt = liftEval T.evalInt

-- | Evaluate an (AST s) node of sort /real/ in the given model.
--
-- See 'modelEval' and 'getReal'.
evalReal :: EvalAst s Rational
evalReal = liftEval T.evalReal

-- | Evaluate an (AST s) node of sort /bit-vector/ in the given model.
--
-- The flag /signed/ decides whether the bit-vector value is
-- interpreted as a signed or unsigned integer.
--
-- See 'modelEval' and 'getBv'.
evalBv :: Bool -- ^ signed?
       -> EvalAst s Integer
evalBv = liftEval . T.evalBv

-- | Evaluate a collection of (AST s) nodes in the given model.
evalT :: (Traversable t) => t (AST s) -> Eval s (t (AST s))
evalT = liftEval T.evalT

-- | Run a evaluation function on a 'Traversable' data structure of '(AST s)'s
-- (e.g. @[(AST s)]@, @Vector (AST s)@, @Maybe (AST s)@, etc).
--
-- This a generic version of 'evalT' which can be used in combination with
-- other helpers. For instance, @mapEval evalInt@ can be used to obtain
-- the 'Integer' interpretation of a list of '(AST s)' of sort /int/.
mapEval :: (Traversable t) => EvalAst s a -> t (AST s) -> Eval s (t a)
mapEval f = traverse f

-- | Get function as a list of argument/value pairs.
evalFunc :: FuncDecl s -> Eval s FuncModel
evalFunc = liftEval T.evalFunc

liftEval :: (r -> a -> m (Maybe b)) -> a -> ReaderT r (MaybeT m) b
liftEval e a = ReaderT $ MaybeT . flip e a

runWithModel :: Eval s a -> Z3 s (Maybe a)
runWithModel (ReaderT f) = runMaybeT $ MaybeT (snd <$> T.getModel) >>= f
