нк
Е
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.14.02unknown8гГ

global_step/Initializer/zerosConst*
value	B	 R *
dtype0	*
_output_shapes
: *
_class
loc:@global_step
k
global_step
VariableV2*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_output_shapes
: *
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
j
input_1Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *b'П*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *b'?*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
Ь
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_class
loc:@dense/kernel*
T0*
_output_shapes

:
*
dtype0
Ю
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@dense/kernel
р
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:
*
_class
loc:@dense/kernel
в
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:
*
_class
loc:@dense/kernel

dense/kernelVarHandleOp*
shared_namedense/kernel*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
shape
:

i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 

dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0*
_class
loc:@dense/kernel

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:
*
_class
loc:@dense/kernel

dense/bias/Initializer/zerosConst*
valueB
*    *
dtype0*
_output_shapes
:
*
_class
loc:@dense/bias


dense/biasVarHandleOp*
shared_name
dense/bias*
_class
loc:@dense/bias*
_output_shapes
: *
dtype0*
shape:

e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
_class
loc:@dense/bias

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
*
_class
loc:@dense/bias
h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:

n
dense/MatMulMatMulinput_1dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

Ѓ
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *7П*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *7?*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
в
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:

*
dtype0
ж
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
ш
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:

*!
_class
loc:@dense_1/kernel
к
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:

*!
_class
loc:@dense_1/kernel

dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0*
shape
:


m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 

dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0*!
_class
loc:@dense_1/kernel

"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

*!
_class
loc:@dense_1/kernel

dense_1/bias/Initializer/zerosConst*
valueB
*    *
dtype0*
_output_shapes
:
*
_class
loc:@dense_1/bias

dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
: *
dtype0*
shape:

i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 

dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0*
_class
loc:@dense_1/bias

 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
*
_class
loc:@dense_1/bias
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:


u
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

W
dense_1/ReluReludense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

Ѓ
/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:*!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *ђъ-П*
dtype0*
_output_shapes
: *!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ђъ-?*
dtype0*
_output_shapes
: *!
_class
loc:@dense_2/kernel
в
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes

:
*
dtype0
ж
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_2/kernel
ш
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:
*!
_class
loc:@dense_2/kernel
к
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:
*!
_class
loc:@dense_2/kernel

dense_2/kernelVarHandleOp*
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
dtype0*
shape
:

m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 

dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0*!
_class
loc:@dense_2/kernel

"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes

:
*!
_class
loc:@dense_2/kernel

dense_2/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@dense_2/bias

dense_2/biasVarHandleOp*
shared_namedense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
: *
dtype0*
shape:
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 

dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0*
_class
loc:@dense_2/bias

 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:*
_class
loc:@dense_2/bias
l
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes

:

w
dense_2/MatMulMatMuldense_1/Reludense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
g
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
|
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
t
)dense_features/PetalLength/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

%dense_features/PetalLength/ExpandDims
ExpandDimsPlaceholder_2)dense_features/PetalLength/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ
u
 dense_features/PetalLength/ShapeShape%dense_features/PetalLength/ExpandDims*
T0*
_output_shapes
:
x
.dense_features/PetalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0dense_features/PetalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0dense_features/PetalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Д
(dense_features/PetalLength/strided_sliceStridedSlice dense_features/PetalLength/Shape.dense_features/PetalLength/strided_slice/stack0dense_features/PetalLength/strided_slice/stack_10dense_features/PetalLength/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
l
*dense_features/PetalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Д
(dense_features/PetalLength/Reshape/shapePack(dense_features/PetalLength/strided_slice*dense_features/PetalLength/Reshape/shape/1*
T0*
N*
_output_shapes
:
А
"dense_features/PetalLength/ReshapeReshape%dense_features/PetalLength/ExpandDims(dense_features/PetalLength/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
s
(dense_features/PetalWidth/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

$dense_features/PetalWidth/ExpandDims
ExpandDimsPlaceholder_3(dense_features/PetalWidth/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ
s
dense_features/PetalWidth/ShapeShape$dense_features/PetalWidth/ExpandDims*
T0*
_output_shapes
:
w
-dense_features/PetalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/dense_features/PetalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/dense_features/PetalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Џ
'dense_features/PetalWidth/strided_sliceStridedSlicedense_features/PetalWidth/Shape-dense_features/PetalWidth/strided_slice/stack/dense_features/PetalWidth/strided_slice/stack_1/dense_features/PetalWidth/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
k
)dense_features/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Б
'dense_features/PetalWidth/Reshape/shapePack'dense_features/PetalWidth/strided_slice)dense_features/PetalWidth/Reshape/shape/1*
T0*
N*
_output_shapes
:
­
!dense_features/PetalWidth/ReshapeReshape$dense_features/PetalWidth/ExpandDims'dense_features/PetalWidth/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
t
)dense_features/SepalLength/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

%dense_features/SepalLength/ExpandDims
ExpandDimsPlaceholder)dense_features/SepalLength/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ
u
 dense_features/SepalLength/ShapeShape%dense_features/SepalLength/ExpandDims*
T0*
_output_shapes
:
x
.dense_features/SepalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
z
0dense_features/SepalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
z
0dense_features/SepalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Д
(dense_features/SepalLength/strided_sliceStridedSlice dense_features/SepalLength/Shape.dense_features/SepalLength/strided_slice/stack0dense_features/SepalLength/strided_slice/stack_10dense_features/SepalLength/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
l
*dense_features/SepalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Д
(dense_features/SepalLength/Reshape/shapePack(dense_features/SepalLength/strided_slice*dense_features/SepalLength/Reshape/shape/1*
T0*
N*
_output_shapes
:
А
"dense_features/SepalLength/ReshapeReshape%dense_features/SepalLength/ExpandDims(dense_features/SepalLength/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
s
(dense_features/SepalWidth/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

$dense_features/SepalWidth/ExpandDims
ExpandDimsPlaceholder_1(dense_features/SepalWidth/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ
s
dense_features/SepalWidth/ShapeShape$dense_features/SepalWidth/ExpandDims*
T0*
_output_shapes
:
w
-dense_features/SepalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/dense_features/SepalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/dense_features/SepalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Џ
'dense_features/SepalWidth/strided_sliceStridedSlicedense_features/SepalWidth/Shape-dense_features/SepalWidth/strided_slice/stack/dense_features/SepalWidth/strided_slice/stack_1/dense_features/SepalWidth/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
k
)dense_features/SepalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Б
'dense_features/SepalWidth/Reshape/shapePack'dense_features/SepalWidth/strided_slice)dense_features/SepalWidth/Reshape/shape/1*
T0*
N*
_output_shapes
:
­
!dense_features/SepalWidth/ReshapeReshape$dense_features/SepalWidth/ExpandDims'dense_features/SepalWidth/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ
e
dense_features/concat/axisConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

dense_features/concatConcatV2"dense_features/PetalLength/Reshape!dense_features/PetalWidth/Reshape"dense_features/SepalLength/Reshape!dense_features/SepalWidth/Reshapedense_features/concat/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ
n
!model/dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:


model/dense/MatMulMatMuldense_features/concat!model/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

i
"model/dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:


model/dense/BiasAddBiasAddmodel/dense/MatMul"model/dense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

_
model/dense/ReluRelumodel/dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

r
#model/dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:



model/dense_1/MatMulMatMulmodel/dense/Relu#model/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

m
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:


model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul$model/dense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

c
model/dense_1/ReluRelumodel/dense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

r
#model/dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes

:


model/dense_2/MatMulMatMulmodel/dense_1/Relu#model/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
m
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:

model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul$model/dense_2/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
g
ArgMaxArgMaxmodel/dense_2/BiasAddArgMax/dimension*
T0*#
_output_shapes
:џџџџџџџџџ
[
SoftmaxSoftmaxmodel/dense_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_b562bce10d5440e7a2a5d426fc20bdfe/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
б
save/SaveV2/tensor_namesConst"/device:CPU:0*v
valuemBkB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBglobal_step*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
ф
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOpglobal_step"/device:CPU:0*
dtypes
	2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
д
save/RestoreV2/tensor_namesConst"/device:CPU:0*v
valuemBkB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBglobal_step*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
Н
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
	2	*0
_output_shapes
:::::::
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
S
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
W
save/AssignVariableOp_1AssignVariableOpdense/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_1/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
W
save/AssignVariableOp_4AssignVariableOpdense_2/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Y
save/AssignVariableOp_5AssignVariableOpdense_2/kernelsave/Identity_6*
dtype0
u
save/AssignAssignglobal_stepsave/RestoreV2:6*
T0	*
_output_shapes
: *
_class
loc:@global_step
Т
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"ц
trainable_variablesЮЫ
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08

dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08"%
saved_model_main_op


group_deps"И
	variablesЊЇ
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08

dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08*г
serving_defaultП
1
PetalLength"
Placeholder_2:0џџџџџџџџџ
0

PetalWidth"
Placeholder_3:0џџџџџџџџџ
/
SepalLength 
Placeholder:0џџџџџџџџџ
0

SepalWidth"
Placeholder_1:0џџџџџџџџџ1
probabilities 
	Softmax:0џџџџџџџџџ&
classes
ArgMax:0	џџџџџџџџџtensorflow/serving/predict