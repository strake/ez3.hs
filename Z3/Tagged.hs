{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MonadComprehensions #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UnicodeSyntax #-}

module Z3.Tagged
  ( -- * Z3 monad
    Z3
  , module Z3.Opts
  , Logic(..)
  , evalZ3
  , evalZ3With
    -- ** Z3 enviroments
  , Z3Env
  , newEnv

  -- * Types
  , Symbol
  , AST
  , Sort
  , FuncDecl
  , App
  , Pattern
  , Constructor
  , Model
  , Base.Context
  , FuncInterp
  , FuncEntry
  , Params
  , Solver
  , SortKind(..)
  , ASTKind(..)
  -- ** Satisfiability result
  , Result(..)

  -- * Parameters
  , mkParams
  , paramsSetBool
  , paramsSetUInt
  , paramsSetDouble
  , paramsSetSymbol
  , paramsToString

  -- * Symbols
  , mkIntSymbol
  , mkStringSymbol

  -- * Sorts
  , mkUninterpretedSort
  , mkBoolSort
  , mkIntSort
  , mkRealSort
  , mkBvSort
  , mkFiniteDomainSort
  , mkArraySort
  , mkTupleSort
  , mkConstructor
  , mkDatatype
  , mkDatatypes
  , mkSetSort

  -- * Constants and Applications
  , mkFuncDecl
  , mkApp
  , mkConst
  , mkFreshConst
  , mkFreshFuncDecl
  -- ** Helpers
  , mkVar
  , mkBoolVar
  , mkRealVar
  , mkIntVar
  , mkBvVar
  , mkFreshVar
  , mkFreshBoolVar
  , mkFreshRealVar
  , mkFreshIntVar
  , mkFreshBvVar

  -- * Propositional Logic and Equality
  , mkTrue
  , mkFalse
  , mkEq
  , mkNot
  , mkIte
  , mkIff
  , mkImplies
  , mkXor
  , mkAnd
  , mkOr
  , mkDistinct
  -- ** Helpers
  , mkBool

  -- * Arithmetic: Integers and Reals
  , mkAdd
  , mkMul
  , mkSub
  , mkUnaryMinus
  , mkDiv
  , mkMod
  , mkRem
  , mkLt
  , mkLe
  , mkGt
  , mkGe
  , mkInt2Real
  , mkReal2Int
  , mkIsInt

  -- * Bit-vectors
  , mkBvnot
  , mkBvredand
  , mkBvredor
  , mkBvand
  , mkBvor
  , mkBvxor
  , mkBvnand
  , mkBvnor
  , mkBvxnor
  , mkBvneg
  , mkBvadd
  , mkBvsub
  , mkBvmul
  , mkBvudiv
  , mkBvsdiv
  , mkBvurem
  , mkBvsrem
  , mkBvsmod
  , mkBvult
  , mkBvslt
  , mkBvule
  , mkBvsle
  , mkBvuge
  , mkBvsge
  , mkBvugt
  , mkBvsgt
  , mkConcat
  , mkExtract
  , mkSignExt
  , mkZeroExt
  , mkRepeat
  , mkBvshl
  , mkBvlshr
  , mkBvashr
  , mkRotateLeft
  , mkRotateRight
  , mkExtRotateLeft
  , mkExtRotateRight
  , mkInt2bv
  , mkBv2int
  , mkBvnegNoOverflow
  , mkBvaddNoOverflow
  , mkBvaddNoUnderflow
  , mkBvsubNoOverflow
  , mkBvsubNoUnderflow
  , mkBvmulNoOverflow
  , mkBvmulNoUnderflow
  , mkBvsdivNoOverflow

  -- * Arrays
  , mkSelect
  , mkStore
  , mkConstArray
  , mkMap
  , mkArrayDefault

  -- * Sets
  , mkEmptySet
  , mkFullSet
  , mkSetAdd
  , mkSetDel
  , mkSetUnion
  , mkSetIntersect
  , mkSetDifference
  , mkSetComplement
  , mkSetMember
  , mkSetSubset

  -- * Numerals
  , mkNumeral
  , mkInt
  , mkReal
  , mkUnsignedInt
  , mkInt64
  , mkUnsignedInt64
  -- ** Helpers
  , mkIntegral
  , mkRational
  , mkFixed
  , mkRealNum
  , mkInteger
  , mkIntNum
  , mkBitvector
  , mkBvNum

  -- * Quantifiers
  , mkPattern
  , mkBound
  , mkForall
  , mkExists
  , mkForallConst
  , mkExistsConst

  -- * Accessors
  , getSymbolString
  , getSortKind
  , getBvSortSize
  , getDatatypeSortConstructors
  , getDatatypeSortRecognizers
  , getDatatypeSortConstructorAccessors
  , getDeclName
  , getArity
  , getDomain
  , getRange
  , appToAst
  , getAppDecl
  , getAppNumArgs
  , getAppArg
  , getAppArgs
  , getSort
  , getArraySortDomain
  , getArraySortRange
  , getBoolValue
  , getAstKind
  , isApp
  , toApp
  , getNumeralString
  , simplify
  , simplifyEx
  , getIndexValue
  , isQuantifierForall
  , isQuantifierExists
  , getQuantifierWeight
  , getQuantifierNumPatterns
  , getQuantifierPatternAST
  , getQuantifierPatterns
  , getQuantifierNumNoPatterns
  , getQuantifierNoPatternAST
  , getQuantifierNoPatterns
  , getQuantifierNumBound
  , getQuantifierBoundName
  , getQuantifierBoundSort
  , getQuantifierBoundVars
  , getQuantifierBody
  -- ** Helpers
  , getBool
  , getInt
  , getReal
  , getBv

  -- * Modifiers
  , substituteVars

  -- * Models
  , modelEval
  , evalArray
  , getConstInterp
  , getFuncInterp
  , hasInterp
  , numConsts
  , numFuncs
  , getConstDecl
  , getFuncDecl
  , getConsts
  , getFuncs
  , isAsArray
  , addFuncInterp
  , addConstInterp
  , getAsArrayFuncDecl
  , funcInterpGetNumEntries
  , funcInterpGetEntry
  , funcInterpGetElse
  , funcInterpGetArity
  , funcEntryGetValue
  , funcEntryGetNumArgs
  , funcEntryGetArg
  , modelToString
  , showModel
  -- ** Helpers
  , EvalAst
  , eval
  , evalBool
  , evalInt
  , evalReal
  , evalBv
  , evalT
  , mapEval
  , FuncModel(..)
  , evalFunc

  -- * Tactics
  , mkTactic
  , andThenTactic
  , orElseTactic
  , skipTactic
  , tryForTactic
  , mkQuantifierEliminationTactic
  , mkAndInverterGraphTactic
  , applyTactic
  , getApplyResultNumSubgoals
  , getApplyResultSubgoal
  , getApplyResultSubgoals
  , mkGoal
  , goalAssert
  , getGoalSize
  , getGoalFormula
  , getGoalFormulas

  -- * String Conversion
  , ASTPrintMode(..)
  , setASTPrintMode
  , astToString
  , patternToString
  , sortToString
  , funcDeclToString
  , benchmarkToSMTLibString

  -- * Parser interface
  , parseSMTLib2String
  , parseSMTLib2File

  -- * Error Handling
  , Base.Z3Error(..)
  , Base.Z3ErrorCode(..)

  -- * Miscellaneous
  , Version(..)
  , getVersion

  -- * Fixedpoint
  , Fixedpoint
  , fixedpointPush
  , fixedpointPop
  , fixedpointAddRule
  , fixedpointSetParams
  , fixedpointRegisterRelation
  , fixedpointQueryRelations
  , fixedpointGetAnswer
  , fixedpointGetAssertions

  -- * Solvers
  , solverGetHelp
  , solverSetParams
  , solverPush
  , solverPop
  , solverReset
  , solverGetNumScopes
  , solverAssertCnstr
  , solverAssertAndTrack
  , solverCheck
  , solverCheckAssumptions
  , solverGetModel
  , solverGetUnsatCore
  , solverGetReasonUnknown
  , solverToString
  -- ** Helpers
  , assert
  , check
  , checkAssumptions
  , solverCheckAndGetModel
  , solverCheckAssumptionsAndGetModel
  , getModel
  , withModel
  , getUnsatCore
  , push
  , pop
  , local
  , reset
  , getNumScopes
  )
  where

import Z3.Opts
import Z3.Base
  ( FuncModel(..)
  , Result(..)
  , Logic(..)
  , ASTPrintMode(..)
  , Version(..)
  , SortKind(..)
  , ASTKind(..)
  )

import qualified Z3.Base as Base

import Control.Monad ((>=>))
import Control.Monad.ST
import Control.Monad.ST.Unsafe
import Control.Monad.Trans.Class ( lift )
import Control.Monad.Trans.Reader ( ReaderT (..) )
import Data.Coerce ( Coercible, coerce )
import Data.Fixed ( Fixed, HasResolution )
import Data.Int ( Int64 )
import Data.List.NonEmpty ( NonEmpty (..) )
import Data.Kind ( Type )
import qualified Data.Traversable as T
import Data.Word ( Word, Word64 )
import Foreign.Storable

liftF0 :: (Coercible a' a)
       => (Base.Context -> IO a') -> Z3 s a
liftF0 f = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context

liftF1 :: (Coercible a' a, Coercible b' b)
       => (Base.Context -> a' -> IO b') -> a -> Z3 s b
liftF1 f a = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context a

liftF2 :: (Coercible a' a, Coercible b' b, Coercible c' c)
       => (Base.Context -> a' -> b' -> IO c') -> a -> b -> Z3 s c
liftF2 f a b = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context a b

liftF3 :: (Coercible a' a, Coercible b' b, Coercible c' c, Coercible d' d)
       => (Base.Context -> a' -> b' -> c' -> IO d') -> a -> b -> c -> Z3 s d
liftF3 f a b c = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context a b c

liftF4 :: (Coercible a' a, Coercible b' b, Coercible c' c, Coercible d' d, Coercible e' e)
       => (Base.Context -> a' -> b' -> c' -> d' -> IO e') -> a -> b -> c -> d -> Z3 s e
liftF4 f a b c d = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context a b c d

liftF5 :: (Coercible a' a, Coercible b' b, Coercible c' c, Coercible d' d, Coercible e' e, Coercible f' f)
       => (Base.Context -> a' -> b' -> c' -> d' -> e' -> IO f') -> a -> b -> c -> d -> e -> Z3 s f
liftF5 f a b c d e = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context a b c d e

liftF6 :: (Coercible a' a, Coercible b' b, Coercible c' c, Coercible d' d, Coercible e' e, Coercible f' f, Coercible g' g)
       => (Base.Context -> a' -> b' -> c' -> d' -> e' -> f' -> IO g') -> a -> b -> c -> d -> e -> f -> Z3 s g
liftF6 φ a b c d e f = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce φ context a b c d e f

liftSolver0 :: (Coercible a' a)
            => (Base.Context -> Base.Solver -> IO a') -> Z3 s a
liftSolver0 f = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context solver

liftSolver1 :: (Coercible a' a, Coercible b' b)
            => (Base.Context -> Base.Solver -> a' -> IO b') -> a -> Z3 s b
liftSolver1 f a = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context solver a

liftSolver2 :: (Coercible a' a, Coercible b' b, Coercible c' c)
            => (Base.Context -> Base.Solver -> a' -> b' -> IO c') -> a -> b -> Z3 s c
liftSolver2 f a b = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context solver a b

liftFixedpoint0 :: (Coercible a' a)
                => (Base.Context -> Base.Fixedpoint -> IO a') -> Z3 s a
liftFixedpoint0 f = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context fixedpoint

liftFixedpoint1 :: (Coercible a' a, Coercible b' b)
                => (Base.Context -> Base.Fixedpoint -> a' -> IO b') -> a -> Z3 s b
liftFixedpoint1 f a = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context fixedpoint a

liftFixedpoint2 :: (Coercible a' a, Coercible b' b, Coercible c' c)
                => (Base.Context -> Base.Fixedpoint -> a' -> b' -> IO c') -> a -> b -> Z3 s c
liftFixedpoint2 f a b = ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ coerce f context fixedpoint a b

-------------------------------------------------
-- A simple Z3 monad.

type Z3 s = ReaderT (Z3Env s) (ST s)

-- | Z3 environment.
data Z3Env s = Z3Env { solver :: Solver s, context :: Context s, fixedpoint :: Fixedpoint s }

-- | Eval a Z3 script.
evalZ3With :: Maybe Logic -> Opts -> Z3 s a -> ST s a
evalZ3With mbLogic opts (ReaderT a) = a =<< newEnv mbLogic opts

-- | Eval a Z3 script with default configuration options.
evalZ3 :: Z3 s a -> ST s a
evalZ3 = evalZ3With Nothing stdOpts


newEnvWith :: (Base.Config -> IO Base.Context) -> Maybe Logic -> Opts -> ST s (Z3Env s)
newEnvWith mkContext mbLogic opts =
  unsafeIOToST . Base.withConfig $ \cfg -> do
    setOpts cfg opts
    ctx <- mkContext cfg
    [coerce Z3Env solver ctx fixedpoint
       | solver <- maybe (Base.mkSolver ctx) (Base.mkSolverForLogic ctx) mbLogic
       , fixedpoint <- Base.mkFixedpoint ctx]

-- | Create a new Z3 environment.
newEnv :: Maybe Logic -> Opts -> ST s (Z3Env s)
newEnv = newEnvWith Base.mkContext

---------------------------------------------------------------------
-- * Parameters

-- | Create a Z3 (empty) parameter set.
--
-- Starting at Z3 4.0, parameter sets are used to configure many components
-- such as: simplifiers, tactics, solvers, etc.
mkParams :: Z3 s (Params s)
mkParams = liftF0 Base.mkParams

-- | Add a Boolean parameter /k/ with value /v/ to the parameter set /p/.
paramsSetBool :: Params s -> Symbol s -> Bool -> Z3 s ()
paramsSetBool = liftF3 Base.paramsSetBool

-- | Add a unsigned parameter /k/ with value /v/ to the parameter set /p/.
paramsSetUInt :: Params s -> Symbol s -> Word -> Z3 s ()
paramsSetUInt = liftF3 Base.paramsSetUInt

-- | Add a double parameter /k/ with value /v/ to the parameter set /p/.
paramsSetDouble :: Params s -> Symbol s -> Double -> Z3 s ()
paramsSetDouble = liftF3 Base.paramsSetDouble

-- | Add a symbol parameter /k/ with value /v/ to the parameter set /p/.
paramsSetSymbol :: Params s -> Symbol s -> Symbol s -> Z3 s ()
paramsSetSymbol = liftF3 Base.paramsSetSymbol

-- | Convert a parameter set into a string.
--
-- This function is mainly used for printing the contents of a parameter set.
paramsToString :: Params s -> Z3 s String
paramsToString = liftF1 Base.paramsToString

-- TODO: Z3_params_validate

---------------------------------------------------------------------
-- Symbols

-- | Create a Z3 symbol using an integer.
mkIntSymbol :: ∀ i s . Integral i => i -> Z3 s (Symbol s)
mkIntSymbol = liftF1 (Base.mkIntSymbol :: _ -> i -> _)

-- | Create a Z3 symbol using a string.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gafebb0d3c212927cf7834c3a20a84ecae>
mkStringSymbol :: String -> Z3 s (Symbol s)
mkStringSymbol = liftF1 Base.mkStringSymbol

---------------------------------------------------------------------
-- Sorts

-- | Create a free (uninterpreted) type using the given name (symbol).
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga736e88741af1c178cbebf94c49aa42de>
mkUninterpretedSort :: Symbol s -> Z3 s (Sort s)
mkUninterpretedSort = liftF1 Base.mkUninterpretedSort

-- | Create the /boolean/ type.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gacdc73510b69a010b71793d429015f342>
mkBoolSort :: Z3 s (Sort s)
mkBoolSort = liftF0 Base.mkBoolSort

-- | Create the /integer/ type.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga6cd426ab5748653b77d389fd3eac1015>
mkIntSort :: Z3 s (Sort s)
mkIntSort = liftF0 Base.mkIntSort

-- | Create the /real/ type.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga40ef93b9738485caed6dc84631c3c1a0>
mkRealSort :: Z3 s (Sort s)
mkRealSort = liftF0 Base.mkRealSort

-- | Create a bit-vector type of the given size.
--
-- This type can also be seen as a machine integer.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaeed000a1bbb84b6ca6fdaac6cf0c1688>
mkBvSort :: ∀ i s . Integral i => i -> Z3 s (Sort s)
mkBvSort = liftF1 (Base.mkBvSort :: _ -> i -> _)

-- | Create a finite domain type.
mkFiniteDomainSort :: Symbol s -> Word64 -> Z3 s (Sort s)
mkFiniteDomainSort = liftF2 Base.mkFiniteDomainSort

-- | Create an array type
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gafe617994cce1b516f46128e448c84445>
--
mkArraySort :: Sort s -> Sort s -> Z3 s (Sort s)
mkArraySort = liftF2 Base.mkArraySort

-- | Create a tuple type
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga7156b9c0a76a28fae46c81f8e3cdf0f1>
mkTupleSort :: Symbol s                          -- ^ Name of the sort
            -> [(Symbol s, Sort s)]                -- ^ Name and sort of each field
            -> Z3 s (Sort s, FuncDecl s, [FuncDecl s]) -- ^ Resulting sort, and function
                                               -- declarations for the
                                               -- constructor and projections.
mkTupleSort = liftF2 Base.mkTupleSort

-- | Create a contructor
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaa779e39f7050b9d51857887954b5f9b0>
mkConstructor :: Symbol s                       -- ^ Name of the sonstructor
              -> Symbol s                       -- ^ Name of recognizer function
              -> [(Symbol s, Maybe (Sort s), Int)]  -- ^ Name, sort option, and sortRefs
              -> Z3 s (Constructor s)
mkConstructor = liftF3 Base.mkConstructor

-- | Create datatype, such as lists, trees, records, enumerations or unions of
--   records. The datatype may be recursive. Return the datatype sort.
--
-- Reference <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gab6809d53327d807da9158abdf75df387>
mkDatatype :: Symbol s
           -> [Constructor s]
           -> Z3 s (Sort s)
mkDatatype = liftF2 Base.mkDatatype

-- | Create mutually recursive datatypes, such as a tree and forest.
--
-- Returns the datatype sorts
mkDatatypes :: [Symbol s]
            -> [[Constructor s]]
            -> Z3 s [Sort s]
mkDatatypes = liftF2 Base.mkDatatypes

-- | Create a set type
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga6865879523e7e882d7e50a2d8445ac8b>
--
mkSetSort :: Sort s -> Z3 s (Sort s)
mkSetSort = liftF1 Base.mkSetSort

---------------------------------------------------------------------
-- Constants and Applications

-- | A Z3 function
mkFuncDecl :: Symbol s -> [Sort s] -> Sort s -> Z3 s (FuncDecl s)
mkFuncDecl = liftF3 Base.mkFuncDecl

-- | Create a constant or function application.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga33a202d86bf628bfab9b6f437536cebe>
mkApp :: FuncDecl s -> [AST s] -> Z3 s (AST s)
mkApp = liftF2 Base.mkApp

-- | Declare and create a constant.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga093c9703393f33ae282ec5e8729354ef>
mkConst :: Symbol s -> Sort s -> Z3 s (AST s)
mkConst = liftF2 Base.mkConst

-- | Declare and create a constant.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga093c9703393f33ae282ec5e8729354ef>
mkFreshConst :: String -> Sort s -> Z3 s (AST s)
mkFreshConst = liftF2 Base.mkFreshConst

-- | Declare a fresh constant or function.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga1f60c7eb41c5603e55a188a14dc929ec>
mkFreshFuncDecl :: String -> [Sort s] -> Sort s -> Z3 s (FuncDecl s)
mkFreshFuncDecl = liftF3 Base.mkFreshFuncDecl

-------------------------------------------------
-- ** Helpers

-- | Declare and create a variable (aka /constant/).
--
-- An alias for 'mkConst'.
mkVar :: Symbol s -> Sort s -> Z3 s (AST s)
mkVar = liftF2 Base.mkVar

-- | Declarate and create a variable of sort /bool/.
--
-- See 'mkVar'.
mkBoolVar :: Symbol s -> Z3 s (AST s)
mkBoolVar = liftF1 Base.mkBoolVar

-- | Declarate and create a variable of sort /real/.
--
-- See 'mkVar'.
mkRealVar :: Symbol s -> Z3 s (AST s)
mkRealVar = liftF1 Base.mkRealVar

-- | Declarate and create a variable of sort /int/.
--
-- See 'mkVar'.
mkIntVar :: Symbol s -> Z3 s (AST s)
mkIntVar = liftF1 Base.mkIntVar

-- | Declarate and create a variable of sort /bit-vector/.
--
-- See 'mkVar'.
mkBvVar :: Symbol s
                   -> Int     -- ^ bit-width
                   -> Z3 s (AST s)
mkBvVar = liftF2 Base.mkBvVar

-- | Declare and create a /fresh/ variable (aka /constant/).
--
-- An alias for 'mkFreshConst'.
mkFreshVar :: String -> Sort s -> Z3 s (AST s)
mkFreshVar = liftF2 Base.mkFreshConst

-- | Declarate and create a /fresh/ variable of sort /bool/.
--
-- See 'mkFreshVar'.
mkFreshBoolVar :: String -> Z3 s (AST s)
mkFreshBoolVar = liftF1 Base.mkFreshBoolVar

-- | Declarate and create a /fresh/ variable of sort /real/.
--
-- See 'mkFreshVar'.
mkFreshRealVar :: String -> Z3 s (AST s)
mkFreshRealVar = liftF1 Base.mkFreshRealVar

-- | Declarate and create a /fresh/ variable of sort /int/.
--
-- See 'mkFreshVar'.
mkFreshIntVar :: String -> Z3 s (AST s)
mkFreshIntVar = liftF1 Base.mkFreshIntVar

-- | Declarate and create a /fresh/ variable of sort /bit-vector/.
--
-- See 'mkFreshVar'.
mkFreshBvVar :: String
                        -> Int     -- ^ bit-width
                        -> Z3 s (AST s)
mkFreshBvVar = liftF2 Base.mkFreshBvVar

---------------------------------------------------------------------
-- Propositional Logic and Equality

-- | Create an (AST s) node representing /true/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gae898e7380409bbc57b56cc5205ef1db7>
mkTrue :: Z3 s (AST s)
mkTrue = liftF0 Base.mkTrue

-- | Create an (AST s) node representing /false/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga5952ac17671117a02001fed6575c778d>
mkFalse :: Z3 s (AST s)
mkFalse = liftF0 Base.mkFalse

-- | Create an (AST s) node representing /l = r/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga95a19ce675b70e22bb0401f7137af37c>
mkEq :: AST s -> AST s -> Z3 s (AST s)
mkEq = liftF2 Base.mkEq

-- | The distinct construct is used for declaring the arguments pairwise
-- distinct.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaa076d3a668e0ec97d61744403153ecf7>
mkDistinct :: NonEmpty (AST s) -> Z3 s (AST s)
mkDistinct = liftF1 Base.mkDistinct

-- | Create an (AST s) node representing /not(a)/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga3329538091996eb7b3dc677760a61072>
mkNot :: AST s -> Z3 s (AST s)
mkNot = liftF1 Base.mkNot

-- | Create an (AST s) node representing an if-then-else: /ite(t1, t2, t3)/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga94417eed5c36e1ad48bcfc8ad6e83547>
mkIte :: AST s -> AST s -> AST s -> Z3 s (AST s)
mkIte = liftF3 Base.mkIte

-- | Create an (AST s) node representing /t1 iff t2/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga930a8e844d345fbebc498ac43a696042>
mkIff :: AST s -> AST s -> Z3 s (AST s)
mkIff = liftF2 Base.mkIff

-- | Create an (AST s) node representing /t1 implies t2/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac829c0e25bbbd30343bf073f7b524517>
mkImplies :: AST s -> AST s -> Z3 s (AST s)
mkImplies = liftF2 Base.mkImplies

-- | Create an (AST s) node representing /t1 xor t2/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gacc6d1b848032dec0c4617b594d4229ec>
mkXor :: AST s -> AST s -> Z3 s (AST s)
mkXor = liftF2 Base.mkXor

-- | Create an (AST s) node representing args[0] and ... and args[num_args-1].
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gacde98ce4a8ed1dde50b9669db4838c61>
mkAnd :: [AST s] -> Z3 s (AST s)
mkAnd = liftF1 Base.mkAnd

-- | Create an (AST s) node representing args[0] or ... or args[num_args-1].
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga00866d16331d505620a6c515302021f9>
mkOr :: [AST s] -> Z3 s (AST s)
mkOr = liftF1 Base.mkOr

-------------------------------------------------
-- ** Helpers

-- | Create an (AST s) node representing the given boolean.
mkBool :: Bool -> Z3 s (AST s)
mkBool = liftF1 Base.mkBool

---------------------------------------------------------------------
-- Arithmetic: Integers and Reals

-- | Create an (AST s) node representing args[0] + ... + args[num_args-1].
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4e4ac0a4e53eee0b4b0ef159ed7d0cd5>
mkAdd :: [AST s] -> Z3 s (AST s)
mkAdd = liftF1 Base.mkAdd

-- | Create an (AST s) node representing args[0] * ... * args[num_args-1].
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gab9affbf8401a18eea474b59ad4adc890>
mkMul :: [AST s] -> Z3 s (AST s)
mkMul = liftF1 Base.mkMul

-- | Create an (AST s) node representing args[0] - ... - args[num_args - 1].
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4f5fea9b683f9e674fd8f14d676cc9a9>
mkSub :: NonEmpty (AST s) -> Z3 s (AST s)
mkSub = liftF1 Base.mkSub

-- | Create an (AST s) node representing -arg.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gadcd2929ad732937e25f34277ce4988ea>
mkUnaryMinus :: AST s -> Z3 s (AST s)
mkUnaryMinus = liftF1 Base.mkUnaryMinus

-- | Create an (AST s) node representing arg1 div arg2.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga1ac60ee8307af8d0b900375914194ff3>
mkDiv :: AST s -> AST s -> Z3 s (AST s)
mkDiv = liftF2 Base.mkDiv

-- | Create an (AST s) node representing arg1 mod arg2.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga8e350ac77e6b8fe805f57efe196e7713>
mkMod :: AST s -> AST s -> Z3 s (AST s)
mkMod = liftF2 Base.mkMod

-- | Create an (AST s) node representing arg1 rem arg2.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga2fcdb17f9039bbdaddf8a30d037bd9ff>
mkRem :: AST s -> AST s -> Z3 s (AST s)
mkRem = liftF2 Base.mkRem

-- | Create less than.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga58a3dc67c5de52cf599c346803ba1534>
mkLt :: AST s -> AST s -> Z3 s (AST s)
mkLt = liftF2 Base.mkLt

-- | Create less than or equal to.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaa9a33d11096841f4e8c407f1578bc0bf>
mkLe :: AST s -> AST s -> Z3 s (AST s)
mkLe = liftF2 Base.mkLe

-- | Create greater than.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga46167b86067586bb742c0557d7babfd3>
mkGt :: AST s -> AST s -> Z3 s (AST s)
mkGt = liftF2 Base.mkGt

-- | Create greater than or equal to.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gad9245cbadb80b192323d01a8360fb942>
mkGe :: AST s -> AST s -> Z3 s (AST s)
mkGe = liftF2 Base.mkGe

-- | Coerce an integer to a real.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga7130641e614c7ebafd28ae16a7681a21>
mkInt2Real :: AST s -> Z3 s (AST s)
mkInt2Real = liftF1 Base.mkInt2Real

-- | Coerce a real to an integer.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga759b6563ba1204aae55289009a3fdc6d>
mkReal2Int :: AST s -> Z3 s (AST s)
mkReal2Int = liftF1 Base.mkReal2Int

-- | Check if a real number is an integer.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaac2ad0fb04e4900fdb4add438d137ad3>
mkIsInt :: AST s -> Z3 s (AST s)
mkIsInt = liftF1 Base.mkIsInt

---------------------------------------------------------------------
-- Bit-vectors

-- | Bitwise negation.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga36cf75c92c54c1ca633a230344f23080>
mkBvnot :: AST s -> Z3 s (AST s)
mkBvnot = liftF1 Base.mkBvnot

-- | Take conjunction of bits in vector, return vector of length 1.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaccc04f2b58903279b1b3be589b00a7d8>
mkBvredand :: AST s -> Z3 s (AST s)
mkBvredand = liftF1 Base.mkBvredand

-- | Take disjunction of bits in vector, return vector of length 1.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gafd18e127c0586abf47ad9cd96895f7d2>
mkBvredor :: AST s -> Z3 s (AST s)
mkBvredor = liftF1 Base.mkBvredor

-- | Bitwise and.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gab96e0ea55334cbcd5a0e79323b57615d>
mkBvand :: AST s -> AST s -> Z3 s (AST s)
mkBvand  = liftF2 Base.mkBvand

-- | Bitwise or.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga77a6ae233fb3371d187c6d559b2843f5>
mkBvor :: AST s -> AST s -> Z3 s (AST s)
mkBvor = liftF2 Base.mkBvor

-- | Bitwise exclusive-or.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga0a3821ea00b1c762205f73e4bc29e7d8>
mkBvxor :: AST s -> AST s -> Z3 s (AST s)
mkBvxor = liftF2 Base.mkBvxor

-- | Bitwise nand.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga96dc37d36efd658fff5b2b4df49b0e61>
mkBvnand :: AST s -> AST s -> Z3 s (AST s)
mkBvnand = liftF2 Base.mkBvnand

-- | Bitwise nor.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gabf15059e9e8a2eafe4929fdfd259aadb>
mkBvnor :: AST s -> AST s -> Z3 s (AST s)
mkBvnor = liftF2 Base.mkBvnor

-- | Bitwise xnor.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga784f5ca36a4b03b93c67242cc94b21d6>
mkBvxnor :: AST s -> AST s -> Z3 s (AST s)
mkBvxnor = liftF2 Base.mkBvxnor

-- | Standard two's complement unary minus.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga0c78be00c03eda4ed6a983224ed5c7b7
mkBvneg :: AST s -> Z3 s (AST s)
mkBvneg = liftF1 Base.mkBvneg

-- | Standard two's complement addition.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga819814e33573f3f9948b32fdc5311158>
mkBvadd :: AST s -> AST s -> Z3 s (AST s)
mkBvadd = liftF2 Base.mkBvadd

-- | Standard two's complement subtraction.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga688c9aa1347888c7a51be4e46c19178e>
mkBvsub :: AST s -> AST s -> Z3 s (AST s)
mkBvsub = liftF2 Base.mkBvsub

-- | Standard two's complement multiplication.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga6abd3dde2a1ceff1704cf7221a72258c>
mkBvmul :: AST s -> AST s -> Z3 s (AST s)
mkBvmul = liftF2 Base.mkBvmul

-- | Unsigned division.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga56ce0cd61666c6f8cf5777286f590544>
mkBvudiv :: AST s -> AST s -> Z3 s (AST s)
mkBvudiv = liftF2 Base.mkBvudiv

-- | Two's complement signed division.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gad240fedb2fda1c1005b8e9d3c7f3d5a0>
mkBvsdiv :: AST s -> AST s -> Z3 s (AST s)
mkBvsdiv = liftF2 Base.mkBvsdiv

-- | Unsigned remainder.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga5df4298ec835e43ddc9e3e0bae690c8d>
mkBvurem :: AST s -> AST s -> Z3 s (AST s)
mkBvurem = liftF2 Base.mkBvurem

-- | Two's complement signed remainder (sign follows dividend).
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga46c18a3042fca174fe659d3185693db1>
mkBvsrem :: AST s -> AST s -> Z3 s (AST s)
mkBvsrem = liftF2 Base.mkBvsrem

-- | Two's complement signed remainder (sign follows divisor).
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga95dac8e6eecb50f63cb82038560e0879>
mkBvsmod :: AST s -> AST s -> Z3 s (AST s)
mkBvsmod = liftF2 Base.mkBvsmod

-- | Unsigned less than.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga5774b22e93abcaf9b594672af6c7c3c4>
mkBvult :: AST s -> AST s -> Z3 s (AST s)
mkBvult = liftF2 Base.mkBvult

-- | Two's complement signed less than.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga8ce08af4ed1fbdf08d4d6e63d171663a>
mkBvslt :: AST s -> AST s -> Z3 s (AST s)
mkBvslt = liftF2 Base.mkBvslt

-- | Unsigned less than or equal to.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gab738b89de0410e70c089d3ac9e696e87>
mkBvule :: AST s -> AST s -> Z3 s (AST s)
mkBvule = liftF2 Base.mkBvule

-- | Two's complement signed less than or equal to.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gab7c026feb93e7d2eab180e96f1e6255d>
mkBvsle :: AST s -> AST s -> Z3 s (AST s)
mkBvsle = liftF2 Base.mkBvsle

-- | Unsigned greater than or equal to.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gade58fbfcf61b67bf8c4a441490d3c4df>
mkBvuge :: AST s -> AST s -> Z3 s (AST s)
mkBvuge = liftF2 Base.mkBvuge

-- | Two's complement signed greater than or equal to.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaeec3414c0e8a90a6aa5a23af36bf6dc5>
mkBvsge :: AST s -> AST s -> Z3 s (AST s)
mkBvsge = liftF2 Base.mkBvsge

-- | Unsigned greater than.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga063ab9f16246c99e5c1c893613927ee3>
mkBvugt :: AST s -> AST s -> Z3 s (AST s)
mkBvugt = liftF2 Base.mkBvugt

-- | Two's complement signed greater than.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4e93a985aa2a7812c7c11a2c65d7c5f0>
mkBvsgt :: AST s -> AST s -> Z3 s (AST s)
mkBvsgt = liftF2 Base.mkBvsgt

-- | Concatenate the given bit-vectors.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gae774128fa5e9ff7458a36bd10e6ca0fa>
mkConcat :: AST s -> AST s -> Z3 s (AST s)
mkConcat = liftF2 Base.mkConcat

-- | Extract the bits high down to low from a bitvector of size m to yield a new
-- bitvector of size /n/, where /n = high - low + 1/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga32d2fe7563f3e6b114c1b97b205d4317>
mkExtract :: Int -> Int -> AST s -> Z3 s (AST s)
mkExtract = liftF3 Base.mkExtract

-- | Sign-extend of the given bit-vector to the (signed) equivalent bitvector
-- of size /m+i/, where /m/ is the size of the given bit-vector.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gad29099270b36d0680bb54b560353c10e>
mkSignExt :: Int -> AST s -> Z3 s (AST s)
mkSignExt = liftF2 Base.mkSignExt

-- | Extend the given bit-vector with zeros to the (unsigned) equivalent
-- bitvector of size /m+i/, where /m/ is the size of the given bit-vector.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac9322fae11365a78640baf9078c428b3>
mkZeroExt :: Int -> AST s -> Z3 s (AST s)
mkZeroExt = liftF2 Base.mkZeroExt

-- | Repeat the given bit-vector up length /i/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga03e81721502ea225c264d1f556c9119d>
mkRepeat :: Int -> AST s -> Z3 s (AST s)
mkRepeat = liftF2 Base.mkRepeat

-- | Shift left.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac8d5e776c786c1172fa0d7dfede454e1>
mkBvshl :: AST s -> AST s -> Z3 s (AST s)
mkBvshl = liftF2 Base.mkBvshl

-- | Logical shift right.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac59645a6edadad79a201f417e4e0c512>
mkBvlshr :: AST s -> AST s -> Z3 s (AST s)
mkBvlshr = liftF2 Base.mkBvlshr

-- | Arithmetic shift right.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga674b580ad605ba1c2c9f9d3748be87c4>
mkBvashr :: AST s -> AST s -> Z3 s (AST s)
mkBvashr = liftF2 Base.mkBvashr

-- | Rotate bits of /t1/ to the left /i/ times.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4932b7d08fea079dd903cd857a52dcda>
mkRotateLeft :: Int -> AST s -> Z3 s (AST s)
mkRotateLeft = liftF2 Base.mkRotateLeft

-- | Rotate bits of /t1/ to the right /i/ times.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga3b94e1bf87ecd1a1858af8ebc1da4a1c>
mkRotateRight :: Int -> AST s -> Z3 s (AST s)
mkRotateRight = liftF2 Base.mkRotateRight

-- | Rotate bits of /t1/ to the left /t2/ times.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaf46f1cb80e5a56044591a76e7c89e5e7>
mkExtRotateLeft :: AST s -> AST s -> Z3 s (AST s)
mkExtRotateLeft = liftF2 Base.mkExtRotateLeft

-- | Rotate bits of /t1/ to the right /t2/ times.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gabb227526c592b523879083f12aab281f>
mkExtRotateRight :: AST s -> AST s -> Z3 s (AST s)
mkExtRotateRight = liftF2 Base.mkExtRotateRight

-- | Create an /n/ bit bit-vector from the integer argument /t1/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga35f89eb05df43fbd9cce7200cc1f30b5>
mkInt2bv :: Int -> AST s -> Z3 s (AST s)
mkInt2bv = liftF2 Base.mkInt2bv

-- | Create an integer from the bit-vector argument /t1/. If /is_signed/ is false,
-- then the bit-vector /t1/ is treated as unsigned. So the result is non-negative
-- and in the range [0..2^/N/-1], where /N/ are the number of bits in /t1/.
-- If /is_signed/ is true, /t1/ is treated as a signed bit-vector.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac87b227dc3821d57258d7f53a28323d4>
mkBv2int :: AST s -> Bool -> Z3 s (AST s)
mkBv2int = liftF2 Base.mkBv2int

-- | Create a predicate that checks that the bit-wise addition of /t1/ and /t2/
-- does not overflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga88f6b5ec876f05e0d7ba51e96c4b077f>
mkBvaddNoOverflow :: AST s -> AST s -> Bool -> Z3 s (AST s)
mkBvaddNoOverflow = liftF3 Base.mkBvaddNoOverflow

-- | Create a predicate that checks that the bit-wise signed addition of /t1/
-- and /t2/ does not underflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga1e2b1927cf4e50000c1600d47a152947>
mkBvaddNoUnderflow :: AST s -> AST s -> Z3 s (AST s)
mkBvaddNoUnderflow = liftF2 Base.mkBvaddNoUnderflow

-- | Create a predicate that checks that the bit-wise signed subtraction of /t1/
-- and /t2/ does not overflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga785f8127b87e0b42130e6d8f52167d7c>
mkBvsubNoOverflow :: AST s -> AST s -> Z3 s (AST s)
mkBvsubNoOverflow = liftF2 Base.mkBvsubNoOverflow

-- | Create a predicate that checks that the bit-wise subtraction of /t1/ and
-- /t2/ does not underflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga6480850f9fa01e14aea936c88ff184c4>
mkBvsubNoUnderflow :: AST s -> AST s -> Z3 s (AST s)
mkBvsubNoUnderflow = liftF2 Base.mkBvsubNoUnderflow

-- | Create a predicate that checks that the bit-wise signed division of /t1/
-- and /t2/ does not overflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaa17e7b2c33dfe2abbd74d390927ae83e>
mkBvsdivNoOverflow :: AST s -> AST s -> Z3 s (AST s)
mkBvsdivNoOverflow = liftF2 Base.mkBvsdivNoOverflow

-- | Check that bit-wise negation does not overflow when /t1/ is interpreted as
-- a signed bit-vector.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gae9c5d72605ddcd0e76657341eaccb6c7>
mkBvnegNoOverflow :: AST s -> Z3 s (AST s)
mkBvnegNoOverflow = liftF1 Base.mkBvnegNoOverflow

-- | Create a predicate that checks that the bit-wise multiplication of /t1/ and
-- /t2/ does not overflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga86f4415719d295a2f6845c70b3aaa1df>
mkBvmulNoOverflow :: AST s -> AST s -> Bool -> Z3 s (AST s)
mkBvmulNoOverflow = liftF3 Base.mkBvmulNoOverflow

-- | Create a predicate that checks that the bit-wise signed multiplication of
-- /t1/ and /t2/ does not underflow.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga501ccc01d737aad3ede5699741717fda>
mkBvmulNoUnderflow :: AST s -> AST s -> Z3 s (AST s)
mkBvmulNoUnderflow = liftF2 Base.mkBvmulNoUnderflow

---------------------------------------------------------------------
-- Arrays

-- | Array read. The argument a is the array and i is the index of the array
-- that gets read.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga38f423f3683379e7f597a7fe59eccb67>
mkSelect :: AST s -> AST s -> Z3 s (AST s)
mkSelect = liftF2 Base.mkSelect

-- | Array update.   
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gae305a4f54b4a64f7e5973ae6ccb13593>
mkStore :: AST s -> AST s -> AST s -> Z3 s (AST s)
mkStore = liftF3 Base.mkStore

-- | Create the constant array.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga84ea6f0c32b99c70033feaa8f00e8f2d>
mkConstArray :: Sort s -> AST s -> Z3 s (AST s)
mkConstArray = liftF2 Base.mkConstArray

-- | map f on the the argument arrays.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga9150242d9430a8c3d55d2ca3b9a4362d>
mkMap :: FuncDecl s -> [AST s] -> Z3 s (AST s)
mkMap = liftF2 Base.mkMap

-- | Access the array default value. Produces the default range value, for
-- arrays that can be represented as finite maps with a default range value.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga78e89cca82f0ab4d5f4e662e5e5fba7d>
mkArrayDefault :: AST s -> Z3 s (AST s)
mkArrayDefault = liftF1 Base.mkArrayDefault

---------------------------------------------------------------------
-- Sets

-- | Create the empty set.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga358b6b80509a567148f1c0ca9252118c>
mkEmptySet :: Sort s -> Z3 s (AST s)
mkEmptySet = liftF1 Base.mkEmptySet

-- | Create the full set.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga5e92662c657374f7332aa32ce4503dd2>
mkFullSet :: Sort s -> Z3 s (AST s)
mkFullSet = liftF1 Base.mkFullSet

-- | Add an element to a set.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga856c3d0e28ce720f53912c2bbdd76175>
mkSetAdd :: AST s -> AST s -> Z3 s (AST s)
mkSetAdd = liftF2 Base.mkSetAdd

-- | Remove an element from a set.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga80e883f39dd3b88f9d0745c8a5b91d1d>
mkSetDel :: AST s -> AST s -> Z3 s (AST s)
mkSetDel = liftF2 Base.mkSetDel

-- | Take the union of a list of sets.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4050162a13d539b8913200963bb4743c>
mkSetUnion :: [AST s] -> Z3 s (AST s)
mkSetUnion = liftF1 Base.mkSetUnion

-- | Take the intersection of a list of sets.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga8a8abff0ebe6aeeaa6c919eaa013049d>
mkSetIntersect :: [AST s] -> Z3 s (AST s)
mkSetIntersect = liftF1 Base.mkSetIntersect

-- | Take the set difference between two sets.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gabb49c62f70b8198362e1a29ba6d8bde1>
mkSetDifference :: AST s -> AST s -> Z3 s (AST s)
mkSetDifference = liftF2 Base.mkSetDifference

-- | Take the complement of a set.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga5c57143c9229cdf730c5103ff696590f>
mkSetComplement :: AST s -> Z3 s (AST s)
mkSetComplement = liftF1 Base.mkSetComplement

-- | Check for set membership.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac6e516f3dce0bdd41095c6d6daf56063>
mkSetMember :: AST s -> AST s -> Z3 s (AST s)
mkSetMember = liftF2 Base.mkSetMember

-- | Check if the first set is a subset of the second set.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga139c5803af0e86464adc7cedc53e7f3a>
mkSetSubset :: AST s -> AST s -> Z3 s (AST s)
mkSetSubset = liftF2 Base.mkSetSubset

---------------------------------------------------------------------
-- * Numerals

-- | Create a numeral of a given sort.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gac8aca397e32ca33618d8024bff32948c>
mkNumeral :: String -> Sort s -> Z3 s (AST s)
mkNumeral = liftF2 Base.mkNumeral

-- | Create a numeral of sort /real/.
mkReal :: Int -> Int -> Z3 s (AST s)
mkReal = liftF2 Base.mkReal

-- | Create a numeral of an int, bit-vector, or finite-domain sort.
--
-- This function can be use to create numerals that fit in a
-- /machine integer/.
-- It is slightly faster than 'mkNumeral' since it is not necessary
-- to parse a string.
mkInt :: Int -> Sort s -> Z3 s (AST s)
mkInt = liftF2 Base.mkInt

-- | Create a numeral of an int, bit-vector, or finite-domain sort.
--
-- This function can be use to create numerals that fit in a
-- /machine unsigned integer/.
-- It is slightly faster than 'mkNumeral' since it is not necessary
-- to parse a string.
mkUnsignedInt :: Word -> Sort s -> Z3 s (AST s)
mkUnsignedInt = liftF2 Base.mkUnsignedInt

-- | Create a numeral of an int, bit-vector, or finite-domain sort.
--
-- This function can be use to create numerals that fit in a
-- /machine 64-bit integer/.
-- It is slightly faster than 'mkNumeral' since it is not necessary
-- to parse a string.
mkInt64 :: Int64 -> Sort s -> Z3 s (AST s)
mkInt64 = liftF2 Base.mkInt64

-- | Create a numeral of an int, bit-vector, or finite-domain sort.
--
-- This function can be use to create numerals that fit in a
-- /machine unsigned 64-bit integer/.
-- It is slightly faster than 'mkNumeral' since it is not necessary
-- to parse a string.
mkUnsignedInt64 :: Word64 -> Sort s -> Z3 s (AST s)
mkUnsignedInt64 = liftF2 Base.mkUnsignedInt64

-------------------------------------------------
-- ** Helpers

-- | Create a numeral of an int, bit-vector, or finite-domain sort.
mkIntegral :: ∀ a s . (Integral a) => a -> Sort s -> Z3 s (AST s)
mkIntegral = liftF2 (Base.mkIntegral :: _ -> a -> _)

-- | Create a numeral of sort /real/ from a 'Rational'.
mkRational :: Rational -> Z3 s (AST s)
mkRational = liftF1 Base.mkRational

-- | Create a numeral of sort /real/ from a 'Fixed'.
mkFixed :: ∀ (a :: Type) s . (HasResolution a) => Fixed a -> Z3 s (AST s)
mkFixed = liftF1 (Base.mkFixed :: _ -> Fixed a -> _)

-- | Create a numeral of sort /real/ from a 'Real'.
mkRealNum :: ∀ r s . (Real r) => r -> Z3 s (AST s)
mkRealNum = liftF1 (Base.mkRealNum :: _ -> r -> _)

-- | Create a numeral of sort /int/ from an 'Integer'.
mkInteger :: Integer -> Z3 s (AST s)
mkInteger = liftF1 Base.mkInteger

-- | Create a numeral of sort /int/ from an 'Integral'.
mkIntNum :: ∀ a s . (Integral a) => a -> Z3 s (AST s)
mkIntNum = liftF1 (Base.mkIntNum :: _ -> a -> _)

-- | Create a numeral of sort /Bit-vector/ from an 'Integer'.
mkBitvector :: Int      -- ^ bit-width
                          -> Integer  -- ^ integer value
                          -> Z3 s (AST s)
mkBitvector = liftF2 Base.mkBitvector

-- | Create a numeral of sort /Bit-vector/ from an 'Integral'.
mkBvNum :: ∀ i s .
           (Integral i) => Int    -- ^ bit-width
                        -> i      -- ^ integer value
                        -> Z3 s (AST s)
mkBvNum = liftF2 (Base.mkBvNum :: _ -> _ -> i -> _)

---------------------------------------------------------------------
-- Quantifiers

mkPattern :: [AST s] -> Z3 s (Pattern s)
mkPattern = liftF1 Base.mkPattern

mkBound :: Int -> Sort s -> Z3 s (AST s)
mkBound = liftF2 Base.mkBound

mkForall :: [Pattern s] -> [Symbol s] -> [Sort s] -> AST s -> Z3 s (AST s)
mkForall = liftF4 Base.mkForall

mkForallConst :: [Pattern s] -> [App s] -> AST s -> Z3 s (AST s)
mkForallConst = liftF3 Base.mkForallConst

mkExistsConst :: [Pattern s] -> [App s] -> AST s -> Z3 s (AST s)
mkExistsConst = liftF3 Base.mkExistsConst

mkExists :: [Pattern s] -> [Symbol s] -> [Sort s] -> AST s -> Z3 s (AST s)
mkExists = liftF4 Base.mkExists

---------------------------------------------------------------------
-- Accessors

-- | Return the symbol name.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaf1683d9464f377e5089ce6ebf2a9bd31>
getSymbolString :: Symbol s -> Z3 s String
getSymbolString = liftF1 Base.getSymbolString

-- | Return the sort kind.
--
-- Reference: <http://z3prover.github.io/api/html/group__capi.html#gacd85d48842c7bfaa696596d16875681a>
getSortKind :: Sort s -> Z3 s SortKind
getSortKind = liftF1 Base.getSortKind

-- | Return the size of the given bit-vector sort.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga8fc3550edace7bc046e16d1f96ddb419>
getBvSortSize :: Sort s -> Z3 s Int
getBvSortSize = liftF1 Base.getBvSortSize

-- | Get list of constructors for datatype.
getDatatypeSortConstructors :: Sort s           -- ^ Datatype sort.
                            -> Z3 s [FuncDecl s]  -- ^ (Constructor s) declarations.
getDatatypeSortConstructors = liftF1 Base.getDatatypeSortConstructors

-- | Get list of recognizers for datatype.
getDatatypeSortRecognizers :: Sort s           -- ^ Datatype sort.
                           -> Z3 s [FuncDecl s]  -- ^ (Constructor s) recognizers.
getDatatypeSortRecognizers = liftF1 Base.getDatatypeSortRecognizers

-- | Get list of accessors for datatype.
getDatatypeSortConstructorAccessors :: Sort s              -- ^ Datatype sort.
                                    -> Z3 s [[FuncDecl s]] -- ^ (Constructor s) recognizers.
getDatatypeSortConstructorAccessors = liftF1 Base.getDatatypeSortConstructorAccessors

-- | Return the constant declaration name as a symbol.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga741b1bf11cb92aa2ec9ef2fef73ff129>
getDeclName :: FuncDecl s -> Z3 s (Symbol s)
getDeclName = liftF1 Base.getDeclName

-- | Returns the number of parameters of the given declaration
getArity :: FuncDecl s -> Z3 s Int
getArity = liftF1 Base.getArity

-- | Returns the sort of the i-th parameter of the given function declaration
getDomain :: FuncDecl s         -- ^ A function declaration
          -> Int              -- ^ i
          -> Z3 s (Sort s)
getDomain = liftF2 Base.getDomain

-- | Returns the range of the given declaration.
getRange :: FuncDecl s -> Z3 s (Sort s)
getRange = liftF1 Base.getRange

-- | Convert an app into (AST s). This is just type casting.
appToAst :: App s -> Z3 s (AST s)
appToAst = liftF1 Base.appToAst

-- | Return the declaration of a constant or function application.
getAppDecl :: App s -> Z3 s (FuncDecl s)
getAppDecl = liftF1 Base.getAppDecl

-- | Return the number of argument of an application. If t is an constant, then the number of arguments is 0.
getAppNumArgs :: App s -> Z3 s Int
getAppNumArgs = liftF1 Base.getAppNumArgs

-- | Return the i-th argument of the given application.
getAppArg :: App s -> Int -> Z3 s (AST s)
getAppArg = liftF2 Base.getAppArg

-- | Return a list of all the arguments of the given application.
getAppArgs :: App s -> Z3 s [AST s]
getAppArgs = liftF1 Base.getAppArgs

-- | Return the sort of an (AST s) node.
getSort :: AST s -> Z3 s (Sort s)
getSort = liftF1 Base.getSort

getArraySortDomain :: Sort s -> Z3 s (Sort s)
getArraySortDomain = liftF1 Base.getArraySortDomain

getArraySortRange :: Sort s -> Z3 s (Sort s)
getArraySortRange = liftF1 Base.getArraySortRange

-- | Returns @Just True@, @Just False@, or @Nothing@ for /undefined/.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga133aaa1ec31af9b570ed7627a3c8c5a4>
getBoolValue :: AST s -> Z3 s (Maybe Bool)
getBoolValue = liftF1 Base.getBoolValue

-- | Return the kind of the given (AST s).
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4c43608feea4cae363ef9c520c239a5c>
getAstKind :: AST s -> Z3 s ASTKind
getAstKind = liftF1 Base.getAstKind

-- | Return True if an ast is APP_AST, False otherwise.
isApp :: AST s -> Z3 s Bool
isApp = liftF1 Base.isApp

-- | Cast (AST s) into an (App s).
toApp :: AST s -> Z3 s (App s)
toApp = liftF1 Base.toApp

-- | Return numeral value, as a string of a numeric constant term.
getNumeralString :: AST s -> Z3 s String
getNumeralString = liftF1 Base.getNumeralString

getIndexValue :: AST s -> Z3 s Int
getIndexValue = liftF1 Base.getIndexValue

isQuantifierForall :: AST s -> Z3 s Bool
isQuantifierForall = liftF1 Base.isQuantifierForall

isQuantifierExists :: AST s -> Z3 s Bool
isQuantifierExists = liftF1 Base.isQuantifierExists

getQuantifierWeight :: AST s -> Z3 s Int
getQuantifierWeight = liftF1 Base.getQuantifierWeight

getQuantifierNumPatterns :: AST s -> Z3 s Int
getQuantifierNumPatterns = liftF1 Base.getQuantifierNumPatterns

getQuantifierPatternAST :: AST s -> Int -> Z3 s (AST s)
getQuantifierPatternAST = liftF2 Base.getQuantifierPatternAST

getQuantifierPatterns :: AST s -> Z3 s [AST s]
getQuantifierPatterns = liftF1 Base.getQuantifierPatterns

getQuantifierNumNoPatterns :: AST s -> Z3 s Int
getQuantifierNumNoPatterns = liftF1 Base.getQuantifierNumNoPatterns

getQuantifierNoPatternAST :: AST s -> Int -> Z3 s (AST s)
getQuantifierNoPatternAST = liftF2 Base.getQuantifierNoPatternAST

getQuantifierNoPatterns :: AST s -> Z3 s [AST s]
getQuantifierNoPatterns = liftF1 Base.getQuantifierNoPatterns

getQuantifierNumBound :: AST s -> Z3 s Int
getQuantifierNumBound = liftF1 Base.getQuantifierNumBound

getQuantifierBoundName :: AST s -> Int -> Z3 s (Symbol s)
getQuantifierBoundName = liftF2 Base.getQuantifierBoundName

getQuantifierBoundSort :: AST s -> Int -> Z3 s (Sort s)
getQuantifierBoundSort = liftF2 Base.getQuantifierBoundSort

getQuantifierBoundVars :: AST s -> Z3 s [AST s]
getQuantifierBoundVars = liftF1 Base.getQuantifierBoundVars

getQuantifierBody :: AST s -> Z3 s (AST s)
getQuantifierBody = liftF1 Base.getQuantifierBody

-- | Simplify the expression.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gada433553406475e5dd6a494ea957844c>
simplify :: AST s -> Z3 s (AST s)
simplify = liftF1 Base.simplify

-- | Simplify the expression using the given parameters.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga34329d4c83ca8c98e18b2884b679008c>
simplifyEx :: AST s -> Params s -> Z3 s (AST s)
simplifyEx = liftF2 Base.simplifyEx

-------------------------------------------------
-- ** Helpers

-- | Read a 'Bool' value from an '(AST s)'
getBool :: AST s -> Z3 s Bool
getBool = liftF1 Base.getBool

-- | Return the integer value
getInt :: AST s -> Z3 s Integer
getInt = liftF1 Base.getInt

-- | Return rational value
getReal :: AST s -> Z3 s Rational
getReal = liftF1 Base.getReal

-- | Read the 'Integer' value from an '(AST s)' of sort /bit-vector/.
--
-- See 'mkBv2int'.
getBv :: AST s
                    -> Bool  -- ^ signed?
                    -> Z3 s Integer
getBv = liftF2 Base.getBv


---------------------------------------------------------------------
-- Modifiers

substituteVars :: AST s -> [AST s] -> Z3 s (AST s)
substituteVars = liftF2 Base.substituteVars

---------------------------------------------------------------------
-- Models

-- | Evaluate an (AST s) node in the given model.
--
-- The evaluation may fail for the following reasons:
--
--     * /t/ contains a quantifier.
--     * the model /m/ is partial.
--     * /t/ is type incorrect.
modelEval :: Model s -> AST s
             -> Bool  -- ^ (Model s) completion?
             -> Z3 s (Maybe (AST s))
modelEval = liftF3 Base.modelEval

-- | Get array as a list of argument/value pairs, if it is
-- represented as a function (ie, using as-array).
evalArray :: Model s -> AST s -> Z3 s (Maybe FuncModel)
evalArray = liftF2 Base.evalArray

getConstInterp :: Model s -> FuncDecl s -> Z3 s (Maybe (AST s))
getConstInterp = liftF2 Base.getConstInterp

-- | Return the interpretation of the function f in the model m.
-- Return NULL, if the model does not assign an interpretation for f.
-- That should be interpreted as: the f does not matter.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gafb9cc5eca9564d8a849c154c5a4a8633>
getFuncInterp :: Model s -> FuncDecl s -> Z3 s (Maybe (FuncInterp s))
getFuncInterp = liftF2 Base.getFuncInterp

hasInterp :: Model s -> FuncDecl s -> Z3 s Bool
hasInterp = liftF2 Base.hasInterp

numConsts :: Model s -> Z3 s Word
numConsts = liftF1 Base.numConsts

numFuncs :: Model s -> Z3 s Word
numFuncs = liftF1 Base.numFuncs

getConstDecl :: Model s -> Word -> Z3 s (FuncDecl s)
getConstDecl = liftF2 Base.getConstDecl

getFuncDecl :: Model s -> Word -> Z3 s (FuncDecl s)
getFuncDecl = liftF2 Base.getFuncDecl

getConsts :: Model s -> Z3 s [FuncDecl s]
getConsts = liftF1 Base.getConsts

getFuncs :: Model s -> Z3 s [FuncDecl s]
getFuncs = liftF1 Base.getFuncs

-- | The (_ as-array f) (AST s) node is a construct for assigning interpretations
-- for arrays in Z3. It is the array such that forall indices i we have that
-- (select (_ as-array f) i) is equal to (f i). This procedure returns Z3_TRUE
-- if the a is an as-array (AST s) node.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga4674da67d226bfb16861829b9f129cfa>
isAsArray :: AST s -> Z3 s Bool
isAsArray = liftF1 Base.isAsArray

addFuncInterp :: Model s -> FuncDecl s -> AST s -> Z3 s (FuncInterp s)
addFuncInterp = liftF3 Base.addFuncInterp

addConstInterp :: Model s -> FuncDecl s -> AST s -> Z3 s ()
addConstInterp = liftF3 Base.addConstInterp


-- | Return the function declaration f associated with a (_ as_array f) node.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga7d9262dc6e79f2aeb23fd4a383589dda>
getAsArrayFuncDecl :: AST s -> Z3 s (FuncDecl s)
getAsArrayFuncDecl = liftF1 Base.getAsArrayFuncDecl

-- | Return the number of entries in the given function interpretation.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga2bab9ae1444940e7593729beec279844>
funcInterpGetNumEntries :: FuncInterp s -> Z3 s Int
funcInterpGetNumEntries = liftF1 Base.funcInterpGetNumEntries

-- | Return a "point" of the given function intepretation.
-- It represents the value of f in a particular point.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaf157e1e1cd8c0cfe6a21be6370f659da>
funcInterpGetEntry :: FuncInterp s -> Int -> Z3 s (FuncEntry s)
funcInterpGetEntry = liftF2 Base.funcInterpGetEntry

-- | Return the 'else' value of the given function interpretation.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga46de7559826ba71b8488d727cba1fb64>
funcInterpGetElse :: FuncInterp s -> Z3 s (AST s)
funcInterpGetElse = liftF1 Base.funcInterpGetElse

-- | Return the arity (number of arguments) of the given function
-- interpretation.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaca22cbdb6f7787aaae5d814f2ab383d8>
funcInterpGetArity :: FuncInterp s -> Z3 s Int
funcInterpGetArity = liftF1 Base.funcInterpGetArity

-- | Return the value of this point.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga9fd65e2ab039aa8e40608c2ecf7084da>
funcEntryGetValue :: FuncEntry s -> Z3 s (AST s)
funcEntryGetValue = liftF1 Base.funcEntryGetValue

-- | Return the number of arguments in a Z3_func_entry object.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga51aed8c5bc4b1f53f0c371312de3ce1a>
funcEntryGetNumArgs :: FuncEntry s -> Z3 s Int
funcEntryGetNumArgs = liftF1 Base.funcEntryGetNumArgs

-- | Return an argument of a Z3_func_entry object.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga6fe03fe3c824fceb52766a4d8c2cbeab>
funcEntryGetArg :: FuncEntry s -> Int -> Z3 s (AST s)
funcEntryGetArg = liftF2 Base.funcEntryGetArg

-- | Convert the given model into a string.
modelToString :: Model s -> Z3 s String
modelToString = liftF1 Base.modelToString

-- | Alias for 'modelToString'.
showModel :: Model s -> Z3 s String
showModel = modelToString

-------------------------------------------------
-- ** Helpers

-- | Type of an evaluation function for '(AST s)'.
--
-- Evaluation may fail (i.e. return 'Nothing') for a few
-- reasons, see 'modelEval'.
type EvalAst s a = Model s -> AST s -> Z3 s (Maybe a)

-- | An alias for 'modelEval' with model completion enabled.
eval :: EvalAst s (AST s)
eval = liftF2 Base.eval

-- | Evaluate an (AST s) node of sort /bool/ in the given model.
--
-- See 'modelEval' and 'getBool'.
evalBool :: EvalAst s Bool
evalBool = liftF2 Base.evalBool

-- | Evaluate an (AST s) node of sort /int/ in the given model.
--
-- See 'modelEval' and 'getInt'.
evalInt :: EvalAst s Integer
evalInt = liftF2 Base.evalInt

-- | Evaluate an (AST s) node of sort /real/ in the given model.
--
-- See 'modelEval' and 'getReal'.
evalReal :: EvalAst s Rational
evalReal = liftF2 Base.evalReal

-- | Evaluate an (AST s) node of sort /bit-vector/ in the given model.
--
-- The flag /signed/ decides whether the bit-vector value is
-- interpreted as a signed or unsigned integer.
--
-- See 'modelEval' and 'getBv'.
evalBv :: Bool -- ^ signed?
                     -> EvalAst s Integer
evalBv = liftF3 Base.evalBv

-- | Evaluate a collection of (AST s) nodes in the given model.
evalT :: ∀ t s . (Traversable t) => Model s -> t (AST s) -> Z3 s (Maybe (t (AST s)))
evalT model asts = (fmap . fmap . fmap) coerce . ReaderT $ \ (Z3Env {..}) -> unsafeIOToST $ Base.evalT (coerce context) (coerce model) (coerce <$> asts)

-- | Run a evaluation function on a 'Traversable' data structure of '(AST s)'s
-- (e.g. @[(AST s)]@, @Vector (AST s)@, @Maybe (AST s)@, etc).
--
-- This a generic version of 'evalT' which can be used in combination with
-- other helpers. For instance, @mapEval evalInt@ can be used to obtain
-- the 'Integer' interpretation of a list of '(AST s)' of sort /int/.
mapEval :: (Traversable t) => EvalAst s a
                           -> Model s
                           -> t (AST s)
                           -> Z3 s (Maybe (t a))
mapEval f m = fmap T.sequence . T.mapM (f m)

-- | Get function as a list of argument/value pairs.
evalFunc :: Model s -> FuncDecl s -> Z3 s (Maybe FuncModel)
evalFunc = liftF2 Base.evalFunc

---------------------------------------------------------------------
-- Tactics

mkTactic :: String -> Z3 s (Tactic s)
mkTactic = liftF1 Base.mkTactic

andThenTactic ::Tactic s ->Tactic s -> Z3 s (Tactic s)
andThenTactic = liftF2 Base.andThenTactic

orElseTactic ::Tactic s ->Tactic s -> Z3 s (Tactic s)
orElseTactic = liftF2 Base.andThenTactic

skipTactic :: Z3 s (Tactic s)
skipTactic = liftF0 Base.skipTactic

tryForTactic ::Tactic s -> Int -> Z3 s (Tactic s)
tryForTactic = liftF2 Base.tryForTactic

mkQuantifierEliminationTactic :: Z3 s (Tactic s)
mkQuantifierEliminationTactic = liftF0 Base.mkQuantifierEliminationTactic

mkAndInverterGraphTactic :: Z3 s (Tactic s)
mkAndInverterGraphTactic = liftF0 Base.mkAndInverterGraphTactic

applyTactic ::Tactic s ->Goal s -> Z3 s( ApplyResult s)
applyTactic = liftF2 Base.applyTactic

getApplyResultNumSubgoals ::ApplyResult s -> Z3 s Int
getApplyResultNumSubgoals = liftF1 Base.getApplyResultNumSubgoals

getApplyResultSubgoal ::ApplyResult s -> Int -> Z3 s (Goal s)
getApplyResultSubgoal = liftF2 Base.getApplyResultSubgoal

getApplyResultSubgoals ::ApplyResult s -> Z3 s [Goal s]
getApplyResultSubgoals = liftF1 Base.getApplyResultSubgoals

mkGoal :: Bool -> Bool -> Bool -> Z3 s( Goal s)
mkGoal = liftF3 Base.mkGoal

goalAssert ::Goal s -> AST s -> Z3 s ()
goalAssert = liftF2 Base.goalAssert

getGoalSize ::Goal s -> Z3 s Int
getGoalSize = liftF1 Base.getGoalSize

getGoalFormula ::Goal s -> Int -> Z3 s (AST s)
getGoalFormula = liftF2 Base.getGoalFormula

getGoalFormulas ::Goal s -> Z3 s [AST s]
getGoalFormulas = liftF1 Base.getGoalFormulas

---------------------------------------------------------------------
-- String Conversion

-- | Set the mode for converting expressions to strings.
setASTPrintMode :: ASTPrintMode -> Z3 s ()
setASTPrintMode = liftF1 Base.setASTPrintMode

-- | Convert an (AST s) to a string.
astToString :: AST s -> Z3 s String
astToString = liftF1 Base.astToString

-- | Convert a pattern to a string.
patternToString :: Pattern s -> Z3 s String
patternToString = liftF1 Base.patternToString

-- | Convert a sort to a string.
sortToString :: Sort s -> Z3 s String
sortToString = liftF1 Base.sortToString

-- | Convert a (FuncDecl s) to a string.
funcDeclToString :: FuncDecl s -> Z3 s String
funcDeclToString = liftF1 Base.funcDeclToString

-- | Convert the given benchmark into SMT-LIB formatted string.
--
-- The output format can be configured via 'setASTPrintMode'.
benchmarkToSMTLibString ::
                               String   -- ^ name
                            -> String   -- ^ logic
                            -> String   -- ^ status
                            -> String   -- ^ attributes
                            -> [AST s]    -- ^ assumptions1
                            -> AST s      -- ^ formula
                            -> Z3 s String
benchmarkToSMTLibString = liftF6 Base.benchmarkToSMTLibString


---------------------------------------------------------------------
-- Parser interface

-- | Parse SMT expressions from a string
--
-- The sort and declaration arguments allow parsing in a context in which variables and functions have already been declared. They are almost never used.
parseSMTLib2String ::
                      String     -- ^ string to parse
                   -> [Symbol s]   -- ^ sort names
                   -> [Sort s]     -- ^ sorts
                   -> [Symbol s]   -- ^ declaration names
                   -> [FuncDecl s] -- ^ declarations
                   -> Z3 s (AST s)
parseSMTLib2String = liftF5 Base.parseSMTLib2String

-- | Parse SMT expressions from a file
--
-- The sort and declaration arguments allow parsing in a context in which variables and functions have already been declared. They are almost never used.
parseSMTLib2File ::
                    String     -- ^ string to parse
                 -> [Symbol s]   -- ^ sort names
                 -> [Sort s]     -- ^ sorts
                 -> [Symbol s]   -- ^ declaration names
                 -> [FuncDecl s] -- ^ declarations
                 -> Z3 s (AST s)
parseSMTLib2File = liftF5 Base.parseSMTLib2File

---------------------------------------------------------------------
-- Miscellaneous

-- | Return Z3 version number information.
getVersion :: Z3 s Version
getVersion = lift . unsafeIOToST $ Base.getVersion

---------------------------------------------------------------------
-- Fixedpoint

fixedpointPush :: Z3 s ()
fixedpointPush = liftFixedpoint0 Base.fixedpointPush

fixedpointPop :: Z3 s ()
fixedpointPop = liftFixedpoint0 Base.fixedpointPush

fixedpointAddRule :: AST s -> Symbol s -> Z3 s ()
fixedpointAddRule = liftFixedpoint2 Base.fixedpointAddRule

fixedpointSetParams :: Params s -> Z3 s ()
fixedpointSetParams = liftFixedpoint1 Base.fixedpointSetParams

fixedpointRegisterRelation :: FuncDecl s -> Z3 s ()
fixedpointRegisterRelation = liftFixedpoint1 Base.fixedpointRegisterRelation

fixedpointQueryRelations :: [FuncDecl s] -> Z3 s Result
fixedpointQueryRelations = liftFixedpoint1 Base.fixedpointQueryRelations

fixedpointGetAnswer :: Z3 s (AST s)
fixedpointGetAnswer = liftFixedpoint0 Base.fixedpointGetAnswer

fixedpointGetAssertions :: Z3 s [AST s]
fixedpointGetAssertions = liftFixedpoint0 Base.fixedpointGetAssertions

---------------------------------------------------------------------
-- * Solvers

-- mkSolver :: Context s -> ST s (Solver s)
-- mkSolver = liftF0 z3_mk_solver

-- mkSimpleSolver :: Context s -> ST s (Solver s)
-- mkSimpleSolver = liftF0 z3_mk_simple_solver

-- mkSolverForLogic :: Context s -> Logic -> ST s (Solver s)
-- mkSolverForLogic c logic = withContextError c $ \cPtr ->
--   do sym <- mkStringSymbol c (show logic)
--      c2h c =<< z3_mk_solver_for_logic cPtr (unSymbol sym)

-- | Return a string describing all solver available parameters.
solverGetHelp :: Z3 s String
solverGetHelp = liftSolver0 Base.solverGetHelp

-- | Set the solver using the given parameters.
solverSetParams :: Params s -> Z3 s ()
solverSetParams = liftSolver1 Base.solverSetParams

-- | Create a backtracking point.
solverPush :: Z3 s ()
solverPush = liftSolver0 Base.solverPush

-- | Backtrack /n/ backtracking points.
solverPop :: Int -> Z3 s ()
solverPop = liftSolver1 Base.solverPop

solverReset :: Z3 s ()
solverReset = liftSolver0 Base.solverReset

-- | Number of backtracking points.
solverGetNumScopes :: Z3 s Int
solverGetNumScopes = liftSolver0 Base.solverGetNumScopes

-- | Assert a constraing into the logical context.
--
-- Reference: <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#ga1a05ff73a564ae7256a2257048a4680a>
solverAssertCnstr :: AST s -> Z3 s ()
solverAssertCnstr = liftSolver1 Base.solverAssertCnstr

-- | Assert a constraint a into the solver, and track it
-- (in the unsat) core using the Boolean constant /p/.
--
-- This API is an alternative to Z3_solver_check_assumptions
-- for extracting unsat cores. Both APIs can be used in the same
-- solver. The unsat core will contain a combination of the Boolean
-- variables provided using Z3_solver_assert_and_track and the
-- Boolean literals provided using Z3_solver_check_assumptions.
solverAssertAndTrack :: AST s -> AST s -> Z3 s ()
solverAssertAndTrack = liftSolver2 Base.solverAssertAndTrack

-- | Check whether the assertions in a given solver are consistent or not.
solverCheck :: Z3 s Result
solverCheck = liftSolver0 Base.solverCheck

-- | Check whether the assertions in the given solver and optional assumptions are consistent or not.
solverCheckAssumptions :: [AST s] -> Z3 s Result
solverCheckAssumptions = liftSolver1 Base.solverCheckAssumptions

-- | Retrieve the model for the last 'solverCheck'.
--
-- The error handler is invoked if a model is not available because
-- the commands above were not invoked for the given solver,
-- or if the result was 'Unsat'.
solverGetModel :: Z3 s (Model s)
solverGetModel = liftSolver0 Base.solverGetModel

-- | Retrieve the unsat core for the last 'solverCheckAssumptions'; the unsat core is a subset of the assumptions
solverGetUnsatCore :: Z3 s [AST s]
solverGetUnsatCore = liftSolver0 Base.solverGetUnsatCore

-- | Return a brief justification for an 'Unknown' result for the commands 'solverCheck' and 'solverCheckAssumptions'.
solverGetReasonUnknown :: Z3 s String
solverGetReasonUnknown = liftSolver0 Base.solverGetReasonUnknown

-- | Convert the given solver into a string.
solverToString :: Z3 s String
solverToString = liftSolver0 Base.solverToString

-------------------------------------------------
-- ** Helpers

-- | Create a backtracking point.
--
-- For @push; m; pop 1@ see 'local'.
push :: Z3 s ()
push = solverPush

-- | Backtrack /n/ backtracking points.
--
-- Contrary to 'solverPop' this funtion checks whether /n/ is within
-- the size of the solver scope stack.
pop :: Int -> Z3 s ()
pop n = do
  scopes <- solverGetNumScopes
  if n <= scopes
    then solverPop n
    else error "Z3.Monad.safePop: too many scopes to backtrack"

-- | Run a query and restore the initial logical context.
--
-- This is a shorthand for 'push', run the query, and 'pop'.
local :: Z3 s a -> Z3 s a
local q = push *> q <* pop 1

-- | Backtrack all the way.
reset :: Z3 s ()
reset = solverReset

-- | Get number of backtracking points.
getNumScopes :: Z3 s Int
getNumScopes = liftSolver0 Base.solverGetNumScopes

assert :: AST s -> Z3 s ()
assert = solverAssertCnstr

-- | Check whether the given logical context is consistent or not.
check :: Z3 s Result
check = solverCheck

-- | Check whether the assertions in the given solver and optional assumptions are consistent or not.
checkAssumptions :: [AST s] -> Z3 s Result
checkAssumptions = solverCheckAssumptions

solverCheckAndGetModel :: Z3 s (Result, Maybe (Model s))
solverCheckAndGetModel = liftSolver0 Base.solverCheckAndGetModel

solverCheckAssumptionsAndGetModel :: [AST s] -> Z3 s (Maybe (Either [AST s] (Model s)))
solverCheckAssumptionsAndGetModel = checkAssumptions >=> \ case
    Undef -> pure Nothing
    Unsat -> Just . Left  <$> getUnsatCore
    Sat   -> Just . Right <$> solverGetModel

-- | Get model.
--
-- Reference : <http://research.microsoft.com/en-us/um/redmond/projects/Z3 s/group__capi.html#gaff310fef80ac8a82d0a51417e073ec0a>
getModel :: Z3 s (Result, Maybe (Model s))
getModel = solverCheckAndGetModel

-- | Check satisfiability and, if /sat/, compute a value from the given model.
--
-- E.g.
-- @
-- withModel $ \\m ->
--   fromJust \<$\> evalInt m x
-- @
withModel :: (Model s -> Z3 s a) -> Z3 s (Result, Maybe a)
withModel f = do
  (r,mb_m) <- getModel
  mb_e <- T.traverse f mb_m
  return (r, mb_e)

-- | Retrieve the unsat core for the last 'checkAssumptions'; the unsat core is a subset of the assumptions.
getUnsatCore :: Z3 s [AST s]
getUnsatCore = solverGetUnsatCore

newtype Config      s = Config      { _unConfig      :: Base.Config      } deriving (Eq)
newtype Context     s = Context     { _unContext     :: Base.Context     } deriving (Eq)
newtype Symbol      s = Symbol      { _unSymbol      :: Base.Symbol      } deriving (Eq, Ord, Show, Storable)
newtype AST         s = AST         { _unAST         :: Base.AST         } deriving (Eq, Ord, Show)
newtype Sort        s = Sort        { _unSort        :: Base.Sort        } deriving (Eq, Ord, Show)
newtype FuncDecl    s = FuncDecl    { _unFuncDecl    :: Base.FuncDecl    } deriving (Eq, Ord, Show)
newtype App         s = App         { _unApp         :: Base.App         } deriving (Eq, Ord, Show)
newtype Pattern     s = Pattern     { _unPattern     :: Base.Pattern     } deriving (Eq, Ord, Show)
newtype Constructor s = Constructor { _unConstructor :: Base.Constructor } deriving (Eq, Ord, Show)
newtype Model       s = Model       { _unModel       :: Base.Model       } deriving (Eq)
newtype FuncInterp  s = FuncInterp  { _unFuncInterp  :: Base.FuncInterp  } deriving (Eq)
newtype FuncEntry   s = FuncEntry   { _unFuncEntry   :: Base.FuncEntry   } deriving (Eq)
newtype Tactic      s = Tactic      { _unTactic      :: Base.Tactic      } deriving (Eq)
newtype Goal        s = Goal        { _unGoal        :: Base.Goal        } deriving (Eq)
newtype ApplyResult s = ApplyResult { _unApplyResult :: Base.ApplyResult } deriving (Eq)
newtype Params      s = Params      { _unParams      :: Base.Params      } deriving (Eq)
newtype Solver      s = Solver      { _unSolver      :: Base.Solver      } deriving (Eq)
newtype Fixedpoint  s = Fixedpoint  { _unFixedpoint  :: Base.Fixedpoint  } deriving (Eq)
