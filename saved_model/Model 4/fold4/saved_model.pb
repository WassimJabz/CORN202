нм
■╧
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

·
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
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
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68У¤
Д
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_18/gamma
Й
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_18/beta
З
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_18/moving_mean
Х
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_18/moving_variance
Э
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
: *
dtype0
Д
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_19/gamma
Й
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_19/beta
З
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_19/moving_mean
Х
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_19/moving_variance
Э
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
: *
dtype0
Д
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_20/gamma
Й
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_20/beta
З
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_20/moving_mean
Х
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_20/moving_variance
Э
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes
: *
dtype0
Д
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0
Р
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_21/gamma
Й
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
:@*
dtype0
О
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_21/beta
З
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_21/moving_mean
Х
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
:@*
dtype0
д
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_21/moving_variance
Э
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
:@*
dtype0
z
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
└А*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_22/gamma
К
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_22/beta
И
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_22/moving_mean
Ц
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_22/moving_variance
Ю
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
_output_shapes	
:А*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_23/gamma
К
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_23/beta
И
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_23/moving_mean
Ц
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_23/moving_variance
Ю
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:А*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	А
*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:
*
dtype0
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
_output_shapes
: *
dtype0	
n
Adamax/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_1
g
!Adamax/beta_1/Read/ReadVariableOpReadVariableOpAdamax/beta_1*
_output_shapes
: *
dtype0
n
Adamax/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_2
g
!Adamax/beta_2/Read/ReadVariableOpReadVariableOpAdamax/beta_2*
_output_shapes
: *
dtype0
l
Adamax/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/decay
e
 Adamax/decay/Read/ReadVariableOpReadVariableOpAdamax/decay*
_output_shapes
: *
dtype0
|
Adamax/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdamax/learning_rate
u
(Adamax/learning_rate/Read/ReadVariableOpReadVariableOpAdamax/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdamax/conv2d_12/kernel/m
П
-Adamax/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_12/kernel/m*&
_output_shapes
: *
dtype0
Ж
Adamax/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv2d_12/bias/m

+Adamax/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_12/bias/m*
_output_shapes
: *
dtype0
в
%Adamax/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adamax/batch_normalization_18/gamma/m
Ы
9Adamax/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_18/gamma/m*
_output_shapes
: *
dtype0
а
$Adamax/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_18/beta/m
Щ
8Adamax/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_18/beta/m*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdamax/conv2d_13/kernel/m
П
-Adamax/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_13/kernel/m*&
_output_shapes
:  *
dtype0
Ж
Adamax/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv2d_13/bias/m

+Adamax/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_13/bias/m*
_output_shapes
: *
dtype0
в
%Adamax/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adamax/batch_normalization_19/gamma/m
Ы
9Adamax/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_19/gamma/m*
_output_shapes
: *
dtype0
а
$Adamax/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_19/beta/m
Щ
8Adamax/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_19/beta/m*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdamax/conv2d_14/kernel/m
П
-Adamax/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_14/kernel/m*&
_output_shapes
:  *
dtype0
Ж
Adamax/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv2d_14/bias/m

+Adamax/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_14/bias/m*
_output_shapes
: *
dtype0
в
%Adamax/batch_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adamax/batch_normalization_20/gamma/m
Ы
9Adamax/batch_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_20/gamma/m*
_output_shapes
: *
dtype0
а
$Adamax/batch_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_20/beta/m
Щ
8Adamax/batch_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_20/beta/m*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdamax/conv2d_15/kernel/m
П
-Adamax/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_15/kernel/m*&
_output_shapes
: @*
dtype0
Ж
Adamax/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdamax/conv2d_15/bias/m

+Adamax/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv2d_15/bias/m*
_output_shapes
:@*
dtype0
в
%Adamax/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adamax/batch_normalization_21/gamma/m
Ы
9Adamax/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_21/gamma/m*
_output_shapes
:@*
dtype0
а
$Adamax/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adamax/batch_normalization_21/beta/m
Щ
8Adamax/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_21/beta/m*
_output_shapes
:@*
dtype0
М
Adamax/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*(
shared_nameAdamax/dense_9/kernel/m
Е
+Adamax/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_9/kernel/m* 
_output_shapes
:
└А*
dtype0
Г
Adamax/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdamax/dense_9/bias/m
|
)Adamax/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_9/bias/m*
_output_shapes	
:А*
dtype0
г
%Adamax/batch_normalization_22/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%Adamax/batch_normalization_22/gamma/m
Ь
9Adamax/batch_normalization_22/gamma/m/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_22/gamma/m*
_output_shapes	
:А*
dtype0
б
$Adamax/batch_normalization_22/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adamax/batch_normalization_22/beta/m
Ъ
8Adamax/batch_normalization_22/beta/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_22/beta/m*
_output_shapes	
:А*
dtype0
О
Adamax/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*)
shared_nameAdamax/dense_10/kernel/m
З
,Adamax/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_10/kernel/m* 
_output_shapes
:
АА*
dtype0
Е
Adamax/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdamax/dense_10/bias/m
~
*Adamax/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_10/bias/m*
_output_shapes	
:А*
dtype0
г
%Adamax/batch_normalization_23/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%Adamax/batch_normalization_23/gamma/m
Ь
9Adamax/batch_normalization_23/gamma/m/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_23/gamma/m*
_output_shapes	
:А*
dtype0
б
$Adamax/batch_normalization_23/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adamax/batch_normalization_23/beta/m
Ъ
8Adamax/batch_normalization_23/beta/m/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_23/beta/m*
_output_shapes	
:А*
dtype0
Н
Adamax/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*)
shared_nameAdamax/dense_11/kernel/m
Ж
,Adamax/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_11/kernel/m*
_output_shapes
:	А
*
dtype0
Д
Adamax/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdamax/dense_11/bias/m
}
*Adamax/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_11/bias/m*
_output_shapes
:
*
dtype0
Ц
Adamax/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdamax/conv2d_12/kernel/v
П
-Adamax/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_12/kernel/v*&
_output_shapes
: *
dtype0
Ж
Adamax/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv2d_12/bias/v

+Adamax/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_12/bias/v*
_output_shapes
: *
dtype0
в
%Adamax/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adamax/batch_normalization_18/gamma/v
Ы
9Adamax/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_18/gamma/v*
_output_shapes
: *
dtype0
а
$Adamax/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_18/beta/v
Щ
8Adamax/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_18/beta/v*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdamax/conv2d_13/kernel/v
П
-Adamax/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_13/kernel/v*&
_output_shapes
:  *
dtype0
Ж
Adamax/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv2d_13/bias/v

+Adamax/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_13/bias/v*
_output_shapes
: *
dtype0
в
%Adamax/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adamax/batch_normalization_19/gamma/v
Ы
9Adamax/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_19/gamma/v*
_output_shapes
: *
dtype0
а
$Adamax/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_19/beta/v
Щ
8Adamax/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_19/beta/v*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  **
shared_nameAdamax/conv2d_14/kernel/v
П
-Adamax/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_14/kernel/v*&
_output_shapes
:  *
dtype0
Ж
Adamax/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv2d_14/bias/v

+Adamax/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_14/bias/v*
_output_shapes
: *
dtype0
в
%Adamax/batch_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adamax/batch_normalization_20/gamma/v
Ы
9Adamax/batch_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_20/gamma/v*
_output_shapes
: *
dtype0
а
$Adamax/batch_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batch_normalization_20/beta/v
Щ
8Adamax/batch_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_20/beta/v*
_output_shapes
: *
dtype0
Ц
Adamax/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdamax/conv2d_15/kernel/v
П
-Adamax/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_15/kernel/v*&
_output_shapes
: @*
dtype0
Ж
Adamax/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdamax/conv2d_15/bias/v

+Adamax/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv2d_15/bias/v*
_output_shapes
:@*
dtype0
в
%Adamax/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adamax/batch_normalization_21/gamma/v
Ы
9Adamax/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_21/gamma/v*
_output_shapes
:@*
dtype0
а
$Adamax/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adamax/batch_normalization_21/beta/v
Щ
8Adamax/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_21/beta/v*
_output_shapes
:@*
dtype0
М
Adamax/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*(
shared_nameAdamax/dense_9/kernel/v
Е
+Adamax/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_9/kernel/v* 
_output_shapes
:
└А*
dtype0
Г
Adamax/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdamax/dense_9/bias/v
|
)Adamax/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_9/bias/v*
_output_shapes	
:А*
dtype0
г
%Adamax/batch_normalization_22/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%Adamax/batch_normalization_22/gamma/v
Ь
9Adamax/batch_normalization_22/gamma/v/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_22/gamma/v*
_output_shapes	
:А*
dtype0
б
$Adamax/batch_normalization_22/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adamax/batch_normalization_22/beta/v
Ъ
8Adamax/batch_normalization_22/beta/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_22/beta/v*
_output_shapes	
:А*
dtype0
О
Adamax/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*)
shared_nameAdamax/dense_10/kernel/v
З
,Adamax/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_10/kernel/v* 
_output_shapes
:
АА*
dtype0
Е
Adamax/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdamax/dense_10/bias/v
~
*Adamax/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_10/bias/v*
_output_shapes	
:А*
dtype0
г
%Adamax/batch_normalization_23/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%Adamax/batch_normalization_23/gamma/v
Ь
9Adamax/batch_normalization_23/gamma/v/Read/ReadVariableOpReadVariableOp%Adamax/batch_normalization_23/gamma/v*
_output_shapes	
:А*
dtype0
б
$Adamax/batch_normalization_23/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$Adamax/batch_normalization_23/beta/v
Ъ
8Adamax/batch_normalization_23/beta/v/Read/ReadVariableOpReadVariableOp$Adamax/batch_normalization_23/beta/v*
_output_shapes	
:А*
dtype0
Н
Adamax/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*)
shared_nameAdamax/dense_11/kernel/v
Ж
,Adamax/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_11/kernel/v*
_output_shapes
:	А
*
dtype0
Д
Adamax/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdamax/dense_11/bias/v
}
*Adamax/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_11/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
т╬
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ь╬
valueС╬BН╬ BЕ╬
╜
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
ж

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
╒
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
ж

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
╒
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
О
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
е
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O_random_generator
P__call__
*Q&call_and_return_all_conditional_losses* 
ж

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
╒
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses*
е
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i_random_generator
j__call__
*k&call_and_return_all_conditional_losses* 
ж

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses*
╒
taxis
	ugamma
vbeta
wmoving_mean
xmoving_variance
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses*
У
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses* 
м
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й_random_generator
К__call__
+Л&call_and_return_all_conditional_losses* 
Ф
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses* 
о
Тkernel
	Уbias
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses*
р
	Ъaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses*
м
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й_random_generator
к__call__
+л&call_and_return_all_conditional_losses* 
о
мkernel
	нbias
о	variables
пtrainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses*
р
	┤axis

╡gamma
	╢beta
╖moving_mean
╕moving_variance
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses*
м
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
├_random_generator
─__call__
+┼&call_and_return_all_conditional_losses* 
о
╞kernel
	╟bias
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
╠__call__
+═&call_and_return_all_conditional_losses*
х
	╬iter
╧beta_1
╨beta_2

╤decay
╥learning_ratem═ m╬(m╧)m╨2m╤3m╥;m╙<m╘Rm╒Sm╓[m╫\m╪lm┘mm┌um█vm▄	Тm▌	Уm▐	Ыm▀	Ьmр	мmс	нmт	╡mу	╢mф	╞mх	╟mцvч vш(vщ)vъ2vы3vь;vэ<vюRvяSvЁ[vё\vЄlvєmvЇuvїvvЎ	Тvў	Уv°	Ыv∙	Ьv·	мv√	нv№	╡v¤	╢v■	╞v 	╟vА*
╕
0
 1
(2
)3
*4
+5
26
37
;8
<9
=10
>11
R12
S13
[14
\15
]16
^17
l18
m19
u20
v21
w22
x23
Т24
У25
Ы26
Ь27
Э28
Ю29
м30
н31
╡32
╢33
╖34
╕35
╞36
╟37*
╘
0
 1
(2
)3
24
35
;6
<7
R8
S9
[10
\11
l12
m13
u14
v15
Т16
У17
Ы18
Ь19
м20
н21
╡22
╢23
╞24
╟25*
* 
╡
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

╪serving_default* 
`Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
Ш
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
▌layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
(0
)1
*2
+3*

(0
)1*
* 
Ш
▐non_trainable_variables
▀layers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 
Ш
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
;0
<1
=2
>3*

;0
<1*
* 
Ш
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
эnon_trainable_variables
юlayers
яmetrics
 Ёlayer_regularization_losses
ёlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ц
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

R0
S1*

R0
S1*
* 
Ш
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
[0
\1
]2
^3*

[0
\1*
* 
Ш
№non_trainable_variables
¤layers
■metrics
  layer_regularization_losses
Аlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ц
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

l0
m1*
* 
Ш
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
u0
v1
w2
x3*

u0
v1*
* 
Ш
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ы
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Т0
У1*

Т0
У1*
* 
Ю
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
Ы0
Ь1
Э2
Ю3*

Ы0
Ь1*
* 
Ю
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEdense_10/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_10/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

м0
н1*

м0
н1*
* 
Ю
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
о	variables
пtrainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEbatch_normalization_23/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_23/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_23/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE&batch_normalization_23/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
╡0
╢1
╖2
╕3*

╡0
╢1*
* 
Ю
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
╣	variables
║trainable_variables
╗regularization_losses
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ь
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
┐	variables
└trainable_variables
┴regularization_losses
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEdense_11/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_11/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

╞0
╟1*

╞0
╟1*
* 
Ю
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
╚	variables
╔trainable_variables
╩regularization_losses
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses*
* 
* 
NH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEAdamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEAdamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
^
*0
+1
=2
>3
]4
^5
w6
x7
Э8
Ю9
╖10
╕11*
в
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20*

┬0
├1*
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

]0
^1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

w0
x1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Э0
Ю1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

╖0
╕1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

─total

┼count
╞	variables
╟	keras_api*
M

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

─0
┼1*

╞	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

╚0
╔1*

╦	variables*
Е
VARIABLE_VALUEAdamax/conv2d_12/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_12/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_18/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_18/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_13/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_13/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_19/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_19/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_14/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_14/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_20/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_20/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_15/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_15/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_21/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_21/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdamax/dense_9/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdamax/dense_9/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_22/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_22/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/dense_10/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/dense_10/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ТЛ
VARIABLE_VALUE%Adamax/batch_normalization_23/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUE$Adamax/batch_normalization_23/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/dense_11/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/dense_11/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_12/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_12/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_18/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_18/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_19/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_19/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_14/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_14/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_20/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_20/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/conv2d_15/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/conv2d_15/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_21/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_21/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdamax/dense_9/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdamax/dense_9/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
СК
VARIABLE_VALUE%Adamax/batch_normalization_22/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE$Adamax/batch_normalization_22/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/dense_10/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/dense_10/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ТЛ
VARIABLE_VALUE%Adamax/batch_normalization_23/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
РЙ
VARIABLE_VALUE$Adamax/batch_normalization_23/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdamax/dense_11/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdamax/dense_11/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Т
serving_default_conv2d_12_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
Щ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_12_inputconv2d_12/kernelconv2d_12/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_9/kerneldense_9/bias&batch_normalization_22/moving_variancebatch_normalization_22/gamma"batch_normalization_22/moving_meanbatch_normalization_22/betadense_10/kerneldense_10/bias&batch_normalization_23/moving_variancebatch_normalization_23/gamma"batch_normalization_23/moving_meanbatch_normalization_23/betadense_11/kerneldense_11/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_349923
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
а(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdamax/iter/Read/ReadVariableOp!Adamax/beta_1/Read/ReadVariableOp!Adamax/beta_2/Read/ReadVariableOp Adamax/decay/Read/ReadVariableOp(Adamax/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp-Adamax/conv2d_12/kernel/m/Read/ReadVariableOp+Adamax/conv2d_12/bias/m/Read/ReadVariableOp9Adamax/batch_normalization_18/gamma/m/Read/ReadVariableOp8Adamax/batch_normalization_18/beta/m/Read/ReadVariableOp-Adamax/conv2d_13/kernel/m/Read/ReadVariableOp+Adamax/conv2d_13/bias/m/Read/ReadVariableOp9Adamax/batch_normalization_19/gamma/m/Read/ReadVariableOp8Adamax/batch_normalization_19/beta/m/Read/ReadVariableOp-Adamax/conv2d_14/kernel/m/Read/ReadVariableOp+Adamax/conv2d_14/bias/m/Read/ReadVariableOp9Adamax/batch_normalization_20/gamma/m/Read/ReadVariableOp8Adamax/batch_normalization_20/beta/m/Read/ReadVariableOp-Adamax/conv2d_15/kernel/m/Read/ReadVariableOp+Adamax/conv2d_15/bias/m/Read/ReadVariableOp9Adamax/batch_normalization_21/gamma/m/Read/ReadVariableOp8Adamax/batch_normalization_21/beta/m/Read/ReadVariableOp+Adamax/dense_9/kernel/m/Read/ReadVariableOp)Adamax/dense_9/bias/m/Read/ReadVariableOp9Adamax/batch_normalization_22/gamma/m/Read/ReadVariableOp8Adamax/batch_normalization_22/beta/m/Read/ReadVariableOp,Adamax/dense_10/kernel/m/Read/ReadVariableOp*Adamax/dense_10/bias/m/Read/ReadVariableOp9Adamax/batch_normalization_23/gamma/m/Read/ReadVariableOp8Adamax/batch_normalization_23/beta/m/Read/ReadVariableOp,Adamax/dense_11/kernel/m/Read/ReadVariableOp*Adamax/dense_11/bias/m/Read/ReadVariableOp-Adamax/conv2d_12/kernel/v/Read/ReadVariableOp+Adamax/conv2d_12/bias/v/Read/ReadVariableOp9Adamax/batch_normalization_18/gamma/v/Read/ReadVariableOp8Adamax/batch_normalization_18/beta/v/Read/ReadVariableOp-Adamax/conv2d_13/kernel/v/Read/ReadVariableOp+Adamax/conv2d_13/bias/v/Read/ReadVariableOp9Adamax/batch_normalization_19/gamma/v/Read/ReadVariableOp8Adamax/batch_normalization_19/beta/v/Read/ReadVariableOp-Adamax/conv2d_14/kernel/v/Read/ReadVariableOp+Adamax/conv2d_14/bias/v/Read/ReadVariableOp9Adamax/batch_normalization_20/gamma/v/Read/ReadVariableOp8Adamax/batch_normalization_20/beta/v/Read/ReadVariableOp-Adamax/conv2d_15/kernel/v/Read/ReadVariableOp+Adamax/conv2d_15/bias/v/Read/ReadVariableOp9Adamax/batch_normalization_21/gamma/v/Read/ReadVariableOp8Adamax/batch_normalization_21/beta/v/Read/ReadVariableOp+Adamax/dense_9/kernel/v/Read/ReadVariableOp)Adamax/dense_9/bias/v/Read/ReadVariableOp9Adamax/batch_normalization_22/gamma/v/Read/ReadVariableOp8Adamax/batch_normalization_22/beta/v/Read/ReadVariableOp,Adamax/dense_10/kernel/v/Read/ReadVariableOp*Adamax/dense_10/bias/v/Read/ReadVariableOp9Adamax/batch_normalization_23/gamma/v/Read/ReadVariableOp8Adamax/batch_normalization_23/beta/v/Read/ReadVariableOp,Adamax/dense_11/kernel/v/Read/ReadVariableOp*Adamax/dense_11/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_350957
▀
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_12/kernelconv2d_12/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_15/kernelconv2d_15/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_9/kerneldense_9/biasbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_variancedense_10/kerneldense_10/biasbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_variancedense_11/kerneldense_11/biasAdamax/iterAdamax/beta_1Adamax/beta_2Adamax/decayAdamax/learning_ratetotalcounttotal_1count_1Adamax/conv2d_12/kernel/mAdamax/conv2d_12/bias/m%Adamax/batch_normalization_18/gamma/m$Adamax/batch_normalization_18/beta/mAdamax/conv2d_13/kernel/mAdamax/conv2d_13/bias/m%Adamax/batch_normalization_19/gamma/m$Adamax/batch_normalization_19/beta/mAdamax/conv2d_14/kernel/mAdamax/conv2d_14/bias/m%Adamax/batch_normalization_20/gamma/m$Adamax/batch_normalization_20/beta/mAdamax/conv2d_15/kernel/mAdamax/conv2d_15/bias/m%Adamax/batch_normalization_21/gamma/m$Adamax/batch_normalization_21/beta/mAdamax/dense_9/kernel/mAdamax/dense_9/bias/m%Adamax/batch_normalization_22/gamma/m$Adamax/batch_normalization_22/beta/mAdamax/dense_10/kernel/mAdamax/dense_10/bias/m%Adamax/batch_normalization_23/gamma/m$Adamax/batch_normalization_23/beta/mAdamax/dense_11/kernel/mAdamax/dense_11/bias/mAdamax/conv2d_12/kernel/vAdamax/conv2d_12/bias/v%Adamax/batch_normalization_18/gamma/v$Adamax/batch_normalization_18/beta/vAdamax/conv2d_13/kernel/vAdamax/conv2d_13/bias/v%Adamax/batch_normalization_19/gamma/v$Adamax/batch_normalization_19/beta/vAdamax/conv2d_14/kernel/vAdamax/conv2d_14/bias/v%Adamax/batch_normalization_20/gamma/v$Adamax/batch_normalization_20/beta/vAdamax/conv2d_15/kernel/vAdamax/conv2d_15/bias/v%Adamax/batch_normalization_21/gamma/v$Adamax/batch_normalization_21/beta/vAdamax/dense_9/kernel/vAdamax/dense_9/bias/v%Adamax/batch_normalization_22/gamma/v$Adamax/batch_normalization_22/beta/vAdamax/dense_10/kernel/vAdamax/dense_10/bias/v%Adamax/batch_normalization_23/gamma/v$Adamax/batch_normalization_23/beta/vAdamax/dense_11/kernel/vAdamax/dense_11/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_351264║Е
я
Я
*__inference_conv2d_13_layer_call_fn_350014

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_348311w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
сh
└
H__inference_sequential_3_layer_call_and_return_conditional_losses_348493

inputs*
conv2d_12_348286: 
conv2d_12_348288: +
batch_normalization_18_348291: +
batch_normalization_18_348293: +
batch_normalization_18_348295: +
batch_normalization_18_348297: *
conv2d_13_348312:  
conv2d_13_348314: +
batch_normalization_19_348317: +
batch_normalization_19_348319: +
batch_normalization_19_348321: +
batch_normalization_19_348323: *
conv2d_14_348346:  
conv2d_14_348348: +
batch_normalization_20_348351: +
batch_normalization_20_348353: +
batch_normalization_20_348355: +
batch_normalization_20_348357: *
conv2d_15_348379: @
conv2d_15_348381:@+
batch_normalization_21_348384:@+
batch_normalization_21_348386:@+
batch_normalization_21_348388:@+
batch_normalization_21_348390:@"
dense_9_348421:
└А
dense_9_348423:	А,
batch_normalization_22_348426:	А,
batch_normalization_22_348428:	А,
batch_normalization_22_348430:	А,
batch_normalization_22_348432:	А#
dense_10_348454:
АА
dense_10_348456:	А,
batch_normalization_23_348459:	А,
batch_normalization_23_348461:	А,
batch_normalization_23_348463:	А,
batch_normalization_23_348465:	А"
dense_11_348487:	А

dense_11_348489:

identityИв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_12/StatefulPartitionedCallв!conv2d_13/StatefulPartitionedCallв!conv2d_14/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв dense_10/StatefulPartitionedCallв dense_11/StatefulPartitionedCallвdense_9/StatefulPartitionedCall 
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_348286conv2d_12_348288*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_348285Щ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_18_348291batch_normalization_18_348293batch_normalization_18_348295batch_normalization_18_348297*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347845░
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_13_348312conv2d_13_348314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_348311Щ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_19_348317batch_normalization_19_348319batch_normalization_19_348321batch_normalization_19_348323*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347909В
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_347960щ
dropout_15/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_348332Ь
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0conv2d_14_348346conv2d_14_348348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_348345Щ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_20_348351batch_normalization_20_348353batch_normalization_20_348355batch_normalization_20_348357*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_347985°
dropout_16/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_348365Ь
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0conv2d_15_348379conv2d_15_348381*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_348378Щ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_21_348384batch_normalization_21_348386batch_normalization_21_348388batch_normalization_21_348390*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348049В
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_348100щ
dropout_17/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_348399█
flatten_3/PartitionedCallPartitionedCall#dropout_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_348407М
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_348421dense_9_348423*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_348420Р
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_22_348426batch_normalization_22_348428batch_normalization_22_348430batch_normalization_22_348432*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348127ё
dropout_18/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_348440С
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_10_348454dense_10_348456*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_348453С
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_23_348459batch_normalization_23_348461batch_normalization_23_348463batch_normalization_23_348465*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348209ё
dropout_19/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_348473Р
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_11_348487dense_11_348489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_348486x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ф
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
░%
я
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_350590

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	АИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ам
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Аъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
х
А	
$__inference_signature_wrapper_349923
conv2d_12_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
└А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А


unknown_36:

identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_347823o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_12_input
я
Я
*__inference_conv2d_14_layer_call_fn_350133

inputs!
unknown:  
	unknown_0: 
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_348345w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┼p
∙
H__inference_sequential_3_layer_call_and_return_conditional_losses_348947

inputs*
conv2d_12_348849: 
conv2d_12_348851: +
batch_normalization_18_348854: +
batch_normalization_18_348856: +
batch_normalization_18_348858: +
batch_normalization_18_348860: *
conv2d_13_348863:  
conv2d_13_348865: +
batch_normalization_19_348868: +
batch_normalization_19_348870: +
batch_normalization_19_348872: +
batch_normalization_19_348874: *
conv2d_14_348879:  
conv2d_14_348881: +
batch_normalization_20_348884: +
batch_normalization_20_348886: +
batch_normalization_20_348888: +
batch_normalization_20_348890: *
conv2d_15_348894: @
conv2d_15_348896:@+
batch_normalization_21_348899:@+
batch_normalization_21_348901:@+
batch_normalization_21_348903:@+
batch_normalization_21_348905:@"
dense_9_348911:
└А
dense_9_348913:	А,
batch_normalization_22_348916:	А,
batch_normalization_22_348918:	А,
batch_normalization_22_348920:	А,
batch_normalization_22_348922:	А#
dense_10_348926:
АА
dense_10_348928:	А,
batch_normalization_23_348931:	А,
batch_normalization_23_348933:	А,
batch_normalization_23_348935:	А,
batch_normalization_23_348937:	А"
dense_11_348941:	А

dense_11_348943:

identityИв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_12/StatefulPartitionedCallв!conv2d_13/StatefulPartitionedCallв!conv2d_14/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв dense_10/StatefulPartitionedCallв dense_11/StatefulPartitionedCallвdense_9/StatefulPartitionedCallв"dropout_15/StatefulPartitionedCallв"dropout_16/StatefulPartitionedCallв"dropout_17/StatefulPartitionedCallв"dropout_18/StatefulPartitionedCallв"dropout_19/StatefulPartitionedCall 
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_348849conv2d_12_348851*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_348285Ч
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_18_348854batch_normalization_18_348856batch_normalization_18_348858batch_normalization_18_348860*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347876░
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_13_348863conv2d_13_348865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_348311Ч
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_19_348868batch_normalization_19_348870batch_normalization_19_348872batch_normalization_19_348874*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347940В
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_347960∙
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_348740д
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0conv2d_14_348879conv2d_14_348881*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_348345Ч
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_20_348884batch_normalization_20_348886batch_normalization_20_348888batch_normalization_20_348890*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_348016н
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_348707д
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0conv2d_15_348894conv2d_15_348896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_348378Ч
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_21_348899batch_normalization_21_348901batch_normalization_21_348903batch_normalization_21_348905*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348080В
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_348100Ю
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_348674у
flatten_3/PartitionedCallPartitionedCall+dropout_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_348407М
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_348911dense_9_348913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_348420О
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_22_348916batch_normalization_22_348918batch_normalization_22_348920batch_normalization_22_348922*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348174ж
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_348635Щ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_10_348926dense_10_348928*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_348453П
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_23_348931batch_normalization_23_348933batch_normalization_23_348935batch_normalization_23_348937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348256ж
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_348602Ш
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_11_348941dense_11_348943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_348486x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Э
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
∙
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_348399

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
∙
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_348365

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:          c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ц
d
+__inference_dropout_16_layer_call_fn_350216

inputs
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_348707w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_15_layer_call_and_return_conditional_losses_350253

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
с
╡
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348127

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
∙
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_350221

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:          c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
│
╓
7__inference_batch_normalization_23_layer_call_fn_350536

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348256p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ф	
╥
7__inference_batch_normalization_21_layer_call_fn_350279

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348080Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_19_layer_call_fn_350038

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347909Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
▌
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_350605

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
з

°
D__inference_dense_10_layer_call_and_return_conditional_losses_350510

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_14_layer_call_and_return_conditional_losses_348345

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
с
╡
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348209

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┤
F
*__inference_flatten_3_layer_call_fn_350357

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_348407a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
│
╓
7__inference_batch_normalization_22_layer_call_fn_350409

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348174p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ї└
Ь"
H__inference_sequential_3_layer_call_and_return_conditional_losses_349627

inputsB
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: <
.batch_normalization_18_readvariableop_resource: >
0batch_normalization_18_readvariableop_1_resource: M
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_13_conv2d_readvariableop_resource:  7
)conv2d_13_biasadd_readvariableop_resource: <
.batch_normalization_19_readvariableop_resource: >
0batch_normalization_19_readvariableop_1_resource: M
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_14_conv2d_readvariableop_resource:  7
)conv2d_14_biasadd_readvariableop_resource: <
.batch_normalization_20_readvariableop_resource: >
0batch_normalization_20_readvariableop_1_resource: M
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_15_conv2d_readvariableop_resource: @7
)conv2d_15_biasadd_readvariableop_resource:@<
.batch_normalization_21_readvariableop_resource:@>
0batch_normalization_21_readvariableop_1_resource:@M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:@:
&dense_9_matmul_readvariableop_resource:
└А6
'dense_9_biasadd_readvariableop_resource:	АG
8batch_normalization_22_batchnorm_readvariableop_resource:	АK
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	АI
:batch_normalization_22_batchnorm_readvariableop_1_resource:	АI
:batch_normalization_22_batchnorm_readvariableop_2_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	АG
8batch_normalization_23_batchnorm_readvariableop_resource:	АK
<batch_normalization_23_batchnorm_mul_readvariableop_resource:	АI
:batch_normalization_23_batchnorm_readvariableop_1_resource:	АI
:batch_normalization_23_batchnorm_readvariableop_2_resource:	А:
'dense_11_matmul_readvariableop_resource:	А
6
(dense_11_biasadd_readvariableop_resource:

identityИв6batch_normalization_18/FusedBatchNormV3/ReadVariableOpв8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_18/ReadVariableOpв'batch_normalization_18/ReadVariableOp_1в6batch_normalization_19/FusedBatchNormV3/ReadVariableOpв8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_19/ReadVariableOpв'batch_normalization_19/ReadVariableOp_1в6batch_normalization_20/FusedBatchNormV3/ReadVariableOpв8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_20/ReadVariableOpв'batch_normalization_20/ReadVariableOp_1в6batch_normalization_21/FusedBatchNormV3/ReadVariableOpв8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_21/ReadVariableOpв'batch_normalization_21/ReadVariableOp_1в/batch_normalization_22/batchnorm/ReadVariableOpв1batch_normalization_22/batchnorm/ReadVariableOp_1в1batch_normalization_22/batchnorm/ReadVariableOp_2в3batch_normalization_22/batchnorm/mul/ReadVariableOpв/batch_normalization_23/batchnorm/ReadVariableOpв1batch_normalization_23/batchnorm/ReadVariableOp_1в1batch_normalization_23/batchnorm/ReadVariableOp_2в3batch_normalization_23/batchnorm/mul/ReadVariableOpв conv2d_12/BiasAdd/ReadVariableOpвconv2d_12/Conv2D/ReadVariableOpв conv2d_13/BiasAdd/ReadVariableOpвconv2d_13/Conv2D/ReadVariableOpв conv2d_14/BiasAdd/ReadVariableOpвconv2d_14/Conv2D/ReadVariableOpв conv2d_15/BiasAdd/ReadVariableOpвconv2d_15/Conv2D/ReadVariableOpвdense_10/BiasAdd/ReadVariableOpвdense_10/MatMul/ReadVariableOpвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpР
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0н
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ж
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          l
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:          Р
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╢
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0┐
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( Р
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0╥
conv2d_13/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ж
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:          Р
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╢
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0┐
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( ╝
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
{
dropout_15/IdentityIdentity max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:          Р
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0├
conv2d_14/Conv2DConv2Ddropout_15/Identity:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ж
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:          Р
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╢
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0┐
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( Ж
dropout_16/IdentityIdentity+batch_normalization_20/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          Р
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0├
conv2d_15/Conv2DConv2Ddropout_16/Identity:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         @Р
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┐
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( ╝
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
{
dropout_17/IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:         @`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  З
flatten_3/ReshapeReshapedropout_17/Identity:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:         └Ж
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0О
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         Ае
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╜
$batch_normalization_22/batchnorm/addAddV27batch_normalization_22/batchnorm/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:Ан
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0║
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аж
&batch_normalization_22/batchnorm/mul_1Muldense_9/Relu:activations:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ай
1batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╕
&batch_normalization_22/batchnorm/mul_2Mul9batch_normalization_22/batchnorm/ReadVariableOp_1:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ай
1batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0╕
$batch_normalization_22/batchnorm/subSub9batch_normalization_22/batchnorm/ReadVariableOp_2:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А╕
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А~
dropout_18/IdentityIdentity*batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         АИ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
dense_10/MatMulMatMuldropout_18/Identity:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         Ае
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0k
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╜
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:Ан
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0║
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аз
&batch_normalization_23/batchnorm/mul_1Muldense_10/Relu:activations:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ай
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╕
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ай
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0╕
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А╕
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А~
dropout_19/IdentityIdentity*batch_normalization_23/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         АЗ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0С
dense_11/MatMulMatMuldropout_19/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Д
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
┘
NoOpNoOp7^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_10^batch_normalization_22/batchnorm/ReadVariableOp2^batch_normalization_22/batchnorm/ReadVariableOp_12^batch_normalization_22/batchnorm/ReadVariableOp_24^batch_normalization_22/batchnorm/mul/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp2^batch_normalization_23/batchnorm/ReadVariableOp_12^batch_normalization_23/batchnorm/ReadVariableOp_24^batch_normalization_23/batchnorm/mul/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2f
1batch_normalization_22/batchnorm/ReadVariableOp_11batch_normalization_22/batchnorm/ReadVariableOp_12f
1batch_normalization_22/batchnorm/ReadVariableOp_21batch_normalization_22/batchnorm/ReadVariableOp_22j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2f
1batch_normalization_23/batchnorm/ReadVariableOp_11batch_normalization_23/batchnorm/ReadVariableOp_12f
1batch_normalization_23/batchnorm/ReadVariableOp_21batch_normalization_23/batchnorm/ReadVariableOp_22j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
∙
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_350112

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:          c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
─
G
+__inference_dropout_15_layer_call_fn_350102

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_348332h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_348016

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
аъ
┌)
!__inference__wrapped_model_347823
conv2d_12_inputO
5sequential_3_conv2d_12_conv2d_readvariableop_resource: D
6sequential_3_conv2d_12_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_18_readvariableop_resource: K
=sequential_3_batch_normalization_18_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_18_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_3_conv2d_13_conv2d_readvariableop_resource:  D
6sequential_3_conv2d_13_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_19_readvariableop_resource: K
=sequential_3_batch_normalization_19_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_19_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_3_conv2d_14_conv2d_readvariableop_resource:  D
6sequential_3_conv2d_14_biasadd_readvariableop_resource: I
;sequential_3_batch_normalization_20_readvariableop_resource: K
=sequential_3_batch_normalization_20_readvariableop_1_resource: Z
Lsequential_3_batch_normalization_20_fusedbatchnormv3_readvariableop_resource: \
Nsequential_3_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_3_conv2d_15_conv2d_readvariableop_resource: @D
6sequential_3_conv2d_15_biasadd_readvariableop_resource:@I
;sequential_3_batch_normalization_21_readvariableop_resource:@K
=sequential_3_batch_normalization_21_readvariableop_1_resource:@Z
Lsequential_3_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_3_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:@G
3sequential_3_dense_9_matmul_readvariableop_resource:
└АC
4sequential_3_dense_9_biasadd_readvariableop_resource:	АT
Esequential_3_batch_normalization_22_batchnorm_readvariableop_resource:	АX
Isequential_3_batch_normalization_22_batchnorm_mul_readvariableop_resource:	АV
Gsequential_3_batch_normalization_22_batchnorm_readvariableop_1_resource:	АV
Gsequential_3_batch_normalization_22_batchnorm_readvariableop_2_resource:	АH
4sequential_3_dense_10_matmul_readvariableop_resource:
ААD
5sequential_3_dense_10_biasadd_readvariableop_resource:	АT
Esequential_3_batch_normalization_23_batchnorm_readvariableop_resource:	АX
Isequential_3_batch_normalization_23_batchnorm_mul_readvariableop_resource:	АV
Gsequential_3_batch_normalization_23_batchnorm_readvariableop_1_resource:	АV
Gsequential_3_batch_normalization_23_batchnorm_readvariableop_2_resource:	АG
4sequential_3_dense_11_matmul_readvariableop_resource:	А
C
5sequential_3_dense_11_biasadd_readvariableop_resource:

identityИвCsequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOpвEsequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1в2sequential_3/batch_normalization_18/ReadVariableOpв4sequential_3/batch_normalization_18/ReadVariableOp_1вCsequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOpвEsequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1в2sequential_3/batch_normalization_19/ReadVariableOpв4sequential_3/batch_normalization_19/ReadVariableOp_1вCsequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOpвEsequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1в2sequential_3/batch_normalization_20/ReadVariableOpв4sequential_3/batch_normalization_20/ReadVariableOp_1вCsequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpвEsequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1в2sequential_3/batch_normalization_21/ReadVariableOpв4sequential_3/batch_normalization_21/ReadVariableOp_1в<sequential_3/batch_normalization_22/batchnorm/ReadVariableOpв>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_1в>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_2в@sequential_3/batch_normalization_22/batchnorm/mul/ReadVariableOpв<sequential_3/batch_normalization_23/batchnorm/ReadVariableOpв>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_1в>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_2в@sequential_3/batch_normalization_23/batchnorm/mul/ReadVariableOpв-sequential_3/conv2d_12/BiasAdd/ReadVariableOpв,sequential_3/conv2d_12/Conv2D/ReadVariableOpв-sequential_3/conv2d_13/BiasAdd/ReadVariableOpв,sequential_3/conv2d_13/Conv2D/ReadVariableOpв-sequential_3/conv2d_14/BiasAdd/ReadVariableOpв,sequential_3/conv2d_14/Conv2D/ReadVariableOpв-sequential_3/conv2d_15/BiasAdd/ReadVariableOpв,sequential_3/conv2d_15/Conv2D/ReadVariableOpв,sequential_3/dense_10/BiasAdd/ReadVariableOpв+sequential_3/dense_10/MatMul/ReadVariableOpв,sequential_3/dense_11/BiasAdd/ReadVariableOpв+sequential_3/dense_11/MatMul/ReadVariableOpв+sequential_3/dense_9/BiasAdd/ReadVariableOpв*sequential_3/dense_9/MatMul/ReadVariableOpк
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╨
sequential_3/conv2d_12/Conv2DConv2Dconv2d_12_input4sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
а
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          Ж
sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:          к
2sequential_3/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_18_readvariableop_resource*
_output_shapes
: *
dtype0о
4sequential_3/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_18_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Csequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Esequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Н
4sequential_3/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3)sequential_3/conv2d_12/Relu:activations:0:sequential_3/batch_normalization_18/ReadVariableOp:value:0<sequential_3/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( к
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0∙
sequential_3/conv2d_13/Conv2DConv2D8sequential_3/batch_normalization_18/FusedBatchNormV3:y:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
а
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          Ж
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:          к
2sequential_3/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_19_readvariableop_resource*
_output_shapes
: *
dtype0о
4sequential_3/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_19_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Csequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Esequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Н
4sequential_3/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3)sequential_3/conv2d_13/Relu:activations:0:sequential_3/batch_normalization_19/ReadVariableOp:value:0<sequential_3/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( ╓
$sequential_3/max_pooling2d_6/MaxPoolMaxPool8sequential_3/batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
Х
 sequential_3/dropout_15/IdentityIdentity-sequential_3/max_pooling2d_6/MaxPool:output:0*
T0*/
_output_shapes
:          к
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ъ
sequential_3/conv2d_14/Conv2DConv2D)sequential_3/dropout_15/Identity:output:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
а
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          Ж
sequential_3/conv2d_14/ReluRelu'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:          к
2sequential_3/batch_normalization_20/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_20_readvariableop_resource*
_output_shapes
: *
dtype0о
4sequential_3/batch_normalization_20/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_20_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
Csequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╨
Esequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Н
4sequential_3/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3)sequential_3/conv2d_14/Relu:activations:0:sequential_3/batch_normalization_20/ReadVariableOp:value:0<sequential_3/batch_normalization_20/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( а
 sequential_3/dropout_16/IdentityIdentity8sequential_3/batch_normalization_20/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          к
,sequential_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ъ
sequential_3/conv2d_15/Conv2DConv2D)sequential_3/dropout_16/Identity:output:04sequential_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
а
-sequential_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┬
sequential_3/conv2d_15/BiasAddBiasAdd&sequential_3/conv2d_15/Conv2D:output:05sequential_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Ж
sequential_3/conv2d_15/ReluRelu'sequential_3/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         @к
2sequential_3/batch_normalization_21/ReadVariableOpReadVariableOp;sequential_3_batch_normalization_21_readvariableop_resource*
_output_shapes
:@*
dtype0о
4sequential_3/batch_normalization_21/ReadVariableOp_1ReadVariableOp=sequential_3_batch_normalization_21_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
Csequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_3_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╨
Esequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_3_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Н
4sequential_3/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3)sequential_3/conv2d_15/Relu:activations:0:sequential_3/batch_normalization_21/ReadVariableOp:value:0<sequential_3/batch_normalization_21/ReadVariableOp_1:value:0Ksequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Msequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( ╓
$sequential_3/max_pooling2d_7/MaxPoolMaxPool8sequential_3/batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Х
 sequential_3/dropout_17/IdentityIdentity-sequential_3/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:         @m
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  о
sequential_3/flatten_3/ReshapeReshape)sequential_3/dropout_17/Identity:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:         └а
*sequential_3/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0╡
sequential_3/dense_9/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_3/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_3/dense_9/BiasAddBiasAdd%sequential_3/dense_9/MatMul:product:03sequential_3/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_3/dense_9/ReluRelu%sequential_3/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А┐
<sequential_3/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOpEsequential_3_batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0x
3sequential_3/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ф
1sequential_3/batch_normalization_22/batchnorm/addAddV2Dsequential_3/batch_normalization_22/batchnorm/ReadVariableOp:value:0<sequential_3/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:АЩ
3sequential_3/batch_normalization_22/batchnorm/RsqrtRsqrt5sequential_3/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:А╟
@sequential_3/batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_3_batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0с
1sequential_3/batch_normalization_22/batchnorm/mulMul7sequential_3/batch_normalization_22/batchnorm/Rsqrt:y:0Hsequential_3/batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А═
3sequential_3/batch_normalization_22/batchnorm/mul_1Mul'sequential_3/dense_9/Relu:activations:05sequential_3/batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А├
>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_3_batch_normalization_22_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▀
3sequential_3/batch_normalization_22/batchnorm/mul_2MulFsequential_3/batch_normalization_22/batchnorm/ReadVariableOp_1:value:05sequential_3/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:А├
>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_3_batch_normalization_22_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0▀
1sequential_3/batch_normalization_22/batchnorm/subSubFsequential_3/batch_normalization_22/batchnorm/ReadVariableOp_2:value:07sequential_3/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А▀
3sequential_3/batch_normalization_22/batchnorm/add_1AddV27sequential_3/batch_normalization_22/batchnorm/mul_1:z:05sequential_3/batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:         АШ
 sequential_3/dropout_18/IdentityIdentity7sequential_3/batch_normalization_22/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         Ав
+sequential_3/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_3/dense_10/MatMulMatMul)sequential_3/dropout_18/Identity:output:03sequential_3/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_3/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_3/dense_10/BiasAddBiasAdd&sequential_3/dense_10/MatMul:product:04sequential_3/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_3/dense_10/ReluRelu&sequential_3/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А┐
<sequential_3/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOpEsequential_3_batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0x
3sequential_3/batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ф
1sequential_3/batch_normalization_23/batchnorm/addAddV2Dsequential_3/batch_normalization_23/batchnorm/ReadVariableOp:value:0<sequential_3/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:АЩ
3sequential_3/batch_normalization_23/batchnorm/RsqrtRsqrt5sequential_3/batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:А╟
@sequential_3/batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_3_batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0с
1sequential_3/batch_normalization_23/batchnorm/mulMul7sequential_3/batch_normalization_23/batchnorm/Rsqrt:y:0Hsequential_3/batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А╬
3sequential_3/batch_normalization_23/batchnorm/mul_1Mul(sequential_3/dense_10/Relu:activations:05sequential_3/batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А├
>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_3_batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▀
3sequential_3/batch_normalization_23/batchnorm/mul_2MulFsequential_3/batch_normalization_23/batchnorm/ReadVariableOp_1:value:05sequential_3/batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:А├
>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_3_batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0▀
1sequential_3/batch_normalization_23/batchnorm/subSubFsequential_3/batch_normalization_23/batchnorm/ReadVariableOp_2:value:07sequential_3/batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А▀
3sequential_3/batch_normalization_23/batchnorm/add_1AddV27sequential_3/batch_normalization_23/batchnorm/mul_1:z:05sequential_3/batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:         АШ
 sequential_3/dropout_19/IdentityIdentity7sequential_3/batch_normalization_23/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         Аб
+sequential_3/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0╕
sequential_3/dense_11/MatMulMatMul)sequential_3/dropout_19/Identity:output:03sequential_3/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Ю
,sequential_3/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╕
sequential_3/dense_11/BiasAddBiasAdd&sequential_3/dense_11/MatMul:product:04sequential_3/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
В
sequential_3/dense_11/SoftmaxSoftmax&sequential_3/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         
v
IdentityIdentity'sequential_3/dense_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
╟
NoOpNoOpD^sequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_18/ReadVariableOp5^sequential_3/batch_normalization_18/ReadVariableOp_1D^sequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_19/ReadVariableOp5^sequential_3/batch_normalization_19/ReadVariableOp_1D^sequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_20/ReadVariableOp5^sequential_3/batch_normalization_20/ReadVariableOp_1D^sequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpF^sequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_13^sequential_3/batch_normalization_21/ReadVariableOp5^sequential_3/batch_normalization_21/ReadVariableOp_1=^sequential_3/batch_normalization_22/batchnorm/ReadVariableOp?^sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_1?^sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_2A^sequential_3/batch_normalization_22/batchnorm/mul/ReadVariableOp=^sequential_3/batch_normalization_23/batchnorm/ReadVariableOp?^sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_1?^sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_2A^sequential_3/batch_normalization_23/batchnorm/mul/ReadVariableOp.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp.^sequential_3/conv2d_15/BiasAdd/ReadVariableOp-^sequential_3/conv2d_15/Conv2D/ReadVariableOp-^sequential_3/dense_10/BiasAdd/ReadVariableOp,^sequential_3/dense_10/MatMul/ReadVariableOp-^sequential_3/dense_11/BiasAdd/ReadVariableOp,^sequential_3/dense_11/MatMul/ReadVariableOp,^sequential_3/dense_9/BiasAdd/ReadVariableOp+^sequential_3/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2К
Csequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2О
Esequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_18/ReadVariableOp2sequential_3/batch_normalization_18/ReadVariableOp2l
4sequential_3/batch_normalization_18/ReadVariableOp_14sequential_3/batch_normalization_18/ReadVariableOp_12К
Csequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2О
Esequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_19/ReadVariableOp2sequential_3/batch_normalization_19/ReadVariableOp2l
4sequential_3/batch_normalization_19/ReadVariableOp_14sequential_3/batch_normalization_19/ReadVariableOp_12К
Csequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2О
Esequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_20/ReadVariableOp2sequential_3/batch_normalization_20/ReadVariableOp2l
4sequential_3/batch_normalization_20/ReadVariableOp_14sequential_3/batch_normalization_20/ReadVariableOp_12К
Csequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOpCsequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2О
Esequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Esequential_3/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12h
2sequential_3/batch_normalization_21/ReadVariableOp2sequential_3/batch_normalization_21/ReadVariableOp2l
4sequential_3/batch_normalization_21/ReadVariableOp_14sequential_3/batch_normalization_21/ReadVariableOp_12|
<sequential_3/batch_normalization_22/batchnorm/ReadVariableOp<sequential_3/batch_normalization_22/batchnorm/ReadVariableOp2А
>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_1>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_12А
>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_2>sequential_3/batch_normalization_22/batchnorm/ReadVariableOp_22Д
@sequential_3/batch_normalization_22/batchnorm/mul/ReadVariableOp@sequential_3/batch_normalization_22/batchnorm/mul/ReadVariableOp2|
<sequential_3/batch_normalization_23/batchnorm/ReadVariableOp<sequential_3/batch_normalization_23/batchnorm/ReadVariableOp2А
>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_1>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_12А
>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_2>sequential_3/batch_normalization_23/batchnorm/ReadVariableOp_22Д
@sequential_3/batch_normalization_23/batchnorm/mul/ReadVariableOp@sequential_3/batch_normalization_23/batchnorm/mul/ReadVariableOp2^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_15/BiasAdd/ReadVariableOp-sequential_3/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_15/Conv2D/ReadVariableOp,sequential_3/conv2d_15/Conv2D/ReadVariableOp2\
,sequential_3/dense_10/BiasAdd/ReadVariableOp,sequential_3/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_10/MatMul/ReadVariableOp+sequential_3/dense_10/MatMul/ReadVariableOp2\
,sequential_3/dense_11/BiasAdd/ReadVariableOp,sequential_3/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_11/MatMul/ReadVariableOp+sequential_3/dense_11/MatMul/ReadVariableOp2Z
+sequential_3/dense_9/BiasAdd/ReadVariableOp+sequential_3/dense_9/BiasAdd/ReadVariableOp2X
*sequential_3/dense_9/MatMul/ReadVariableOp*sequential_3/dense_9/MatMul/ReadVariableOp:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_12_input
╟
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_350363

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_349987

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_21_layer_call_fn_350266

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348049Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╡
╓
7__inference_batch_normalization_22_layer_call_fn_350396

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348127p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─
G
+__inference_dropout_16_layer_call_fn_350211

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_348365h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
и
G
+__inference_dropout_18_layer_call_fn_350468

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_348440a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
¤├
╠.
__inference__traced_save_350957
file_prefix/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop*
&savev2_adamax_iter_read_readvariableop	,
(savev2_adamax_beta_1_read_readvariableop,
(savev2_adamax_beta_2_read_readvariableop+
'savev2_adamax_decay_read_readvariableop3
/savev2_adamax_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop8
4savev2_adamax_conv2d_12_kernel_m_read_readvariableop6
2savev2_adamax_conv2d_12_bias_m_read_readvariableopD
@savev2_adamax_batch_normalization_18_gamma_m_read_readvariableopC
?savev2_adamax_batch_normalization_18_beta_m_read_readvariableop8
4savev2_adamax_conv2d_13_kernel_m_read_readvariableop6
2savev2_adamax_conv2d_13_bias_m_read_readvariableopD
@savev2_adamax_batch_normalization_19_gamma_m_read_readvariableopC
?savev2_adamax_batch_normalization_19_beta_m_read_readvariableop8
4savev2_adamax_conv2d_14_kernel_m_read_readvariableop6
2savev2_adamax_conv2d_14_bias_m_read_readvariableopD
@savev2_adamax_batch_normalization_20_gamma_m_read_readvariableopC
?savev2_adamax_batch_normalization_20_beta_m_read_readvariableop8
4savev2_adamax_conv2d_15_kernel_m_read_readvariableop6
2savev2_adamax_conv2d_15_bias_m_read_readvariableopD
@savev2_adamax_batch_normalization_21_gamma_m_read_readvariableopC
?savev2_adamax_batch_normalization_21_beta_m_read_readvariableop6
2savev2_adamax_dense_9_kernel_m_read_readvariableop4
0savev2_adamax_dense_9_bias_m_read_readvariableopD
@savev2_adamax_batch_normalization_22_gamma_m_read_readvariableopC
?savev2_adamax_batch_normalization_22_beta_m_read_readvariableop7
3savev2_adamax_dense_10_kernel_m_read_readvariableop5
1savev2_adamax_dense_10_bias_m_read_readvariableopD
@savev2_adamax_batch_normalization_23_gamma_m_read_readvariableopC
?savev2_adamax_batch_normalization_23_beta_m_read_readvariableop7
3savev2_adamax_dense_11_kernel_m_read_readvariableop5
1savev2_adamax_dense_11_bias_m_read_readvariableop8
4savev2_adamax_conv2d_12_kernel_v_read_readvariableop6
2savev2_adamax_conv2d_12_bias_v_read_readvariableopD
@savev2_adamax_batch_normalization_18_gamma_v_read_readvariableopC
?savev2_adamax_batch_normalization_18_beta_v_read_readvariableop8
4savev2_adamax_conv2d_13_kernel_v_read_readvariableop6
2savev2_adamax_conv2d_13_bias_v_read_readvariableopD
@savev2_adamax_batch_normalization_19_gamma_v_read_readvariableopC
?savev2_adamax_batch_normalization_19_beta_v_read_readvariableop8
4savev2_adamax_conv2d_14_kernel_v_read_readvariableop6
2savev2_adamax_conv2d_14_bias_v_read_readvariableopD
@savev2_adamax_batch_normalization_20_gamma_v_read_readvariableopC
?savev2_adamax_batch_normalization_20_beta_v_read_readvariableop8
4savev2_adamax_conv2d_15_kernel_v_read_readvariableop6
2savev2_adamax_conv2d_15_bias_v_read_readvariableopD
@savev2_adamax_batch_normalization_21_gamma_v_read_readvariableopC
?savev2_adamax_batch_normalization_21_beta_v_read_readvariableop6
2savev2_adamax_dense_9_kernel_v_read_readvariableop4
0savev2_adamax_dense_9_bias_v_read_readvariableopD
@savev2_adamax_batch_normalization_22_gamma_v_read_readvariableopC
?savev2_adamax_batch_normalization_22_beta_v_read_readvariableop7
3savev2_adamax_dense_10_kernel_v_read_readvariableop5
1savev2_adamax_dense_10_bias_v_read_readvariableopD
@savev2_adamax_batch_normalization_23_gamma_v_read_readvariableopC
?savev2_adamax_batch_normalization_23_beta_v_read_readvariableop7
3savev2_adamax_dense_11_kernel_v_read_readvariableop5
1savev2_adamax_dense_11_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: п7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*╪6
value╬6B╦6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╕
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*▌
value╙B╨dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ч,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop&savev2_adamax_iter_read_readvariableop(savev2_adamax_beta_1_read_readvariableop(savev2_adamax_beta_2_read_readvariableop'savev2_adamax_decay_read_readvariableop/savev2_adamax_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop4savev2_adamax_conv2d_12_kernel_m_read_readvariableop2savev2_adamax_conv2d_12_bias_m_read_readvariableop@savev2_adamax_batch_normalization_18_gamma_m_read_readvariableop?savev2_adamax_batch_normalization_18_beta_m_read_readvariableop4savev2_adamax_conv2d_13_kernel_m_read_readvariableop2savev2_adamax_conv2d_13_bias_m_read_readvariableop@savev2_adamax_batch_normalization_19_gamma_m_read_readvariableop?savev2_adamax_batch_normalization_19_beta_m_read_readvariableop4savev2_adamax_conv2d_14_kernel_m_read_readvariableop2savev2_adamax_conv2d_14_bias_m_read_readvariableop@savev2_adamax_batch_normalization_20_gamma_m_read_readvariableop?savev2_adamax_batch_normalization_20_beta_m_read_readvariableop4savev2_adamax_conv2d_15_kernel_m_read_readvariableop2savev2_adamax_conv2d_15_bias_m_read_readvariableop@savev2_adamax_batch_normalization_21_gamma_m_read_readvariableop?savev2_adamax_batch_normalization_21_beta_m_read_readvariableop2savev2_adamax_dense_9_kernel_m_read_readvariableop0savev2_adamax_dense_9_bias_m_read_readvariableop@savev2_adamax_batch_normalization_22_gamma_m_read_readvariableop?savev2_adamax_batch_normalization_22_beta_m_read_readvariableop3savev2_adamax_dense_10_kernel_m_read_readvariableop1savev2_adamax_dense_10_bias_m_read_readvariableop@savev2_adamax_batch_normalization_23_gamma_m_read_readvariableop?savev2_adamax_batch_normalization_23_beta_m_read_readvariableop3savev2_adamax_dense_11_kernel_m_read_readvariableop1savev2_adamax_dense_11_bias_m_read_readvariableop4savev2_adamax_conv2d_12_kernel_v_read_readvariableop2savev2_adamax_conv2d_12_bias_v_read_readvariableop@savev2_adamax_batch_normalization_18_gamma_v_read_readvariableop?savev2_adamax_batch_normalization_18_beta_v_read_readvariableop4savev2_adamax_conv2d_13_kernel_v_read_readvariableop2savev2_adamax_conv2d_13_bias_v_read_readvariableop@savev2_adamax_batch_normalization_19_gamma_v_read_readvariableop?savev2_adamax_batch_normalization_19_beta_v_read_readvariableop4savev2_adamax_conv2d_14_kernel_v_read_readvariableop2savev2_adamax_conv2d_14_bias_v_read_readvariableop@savev2_adamax_batch_normalization_20_gamma_v_read_readvariableop?savev2_adamax_batch_normalization_20_beta_v_read_readvariableop4savev2_adamax_conv2d_15_kernel_v_read_readvariableop2savev2_adamax_conv2d_15_bias_v_read_readvariableop@savev2_adamax_batch_normalization_21_gamma_v_read_readvariableop?savev2_adamax_batch_normalization_21_beta_v_read_readvariableop2savev2_adamax_dense_9_kernel_v_read_readvariableop0savev2_adamax_dense_9_bias_v_read_readvariableop@savev2_adamax_batch_normalization_22_gamma_v_read_readvariableop?savev2_adamax_batch_normalization_22_beta_v_read_readvariableop3savev2_adamax_dense_10_kernel_v_read_readvariableop1savev2_adamax_dense_10_bias_v_read_readvariableop@savev2_adamax_batch_normalization_23_gamma_v_read_readvariableop?savev2_adamax_batch_normalization_23_beta_v_read_readvariableop3savev2_adamax_dense_11_kernel_v_read_readvariableop1savev2_adamax_dense_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*а
_input_shapesО
Л: : : : : : : :  : : : : : :  : : : : : : @:@:@:@:@:@:
└А:А:А:А:А:А:
АА:А:А:А:А:А:	А
:
: : : : : : : : : : : : : :  : : : :  : : : : @:@:@:@:
└А:А:А:А:
АА:А:А:А:	А
:
: : : : :  : : : :  : : : : @:@:@:@:
└А:А:А:А:
АА:А:А:А:	А
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
└А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:! 

_output_shapes	
:А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:%%!

_output_shapes
:	А
: &

_output_shapes
:
:'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
:  : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
:  : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: :,<(
&
_output_shapes
: @: =

_output_shapes
:@: >

_output_shapes
:@: ?

_output_shapes
:@:&@"
 
_output_shapes
:
└А:!A

_output_shapes	
:А:!B

_output_shapes	
:А:!C

_output_shapes	
:А:&D"
 
_output_shapes
:
АА:!E

_output_shapes	
:А:!F

_output_shapes	
:А:!G

_output_shapes	
:А:%H!

_output_shapes
:	А
: I

_output_shapes
:
:,J(
&
_output_shapes
: : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
:  : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :,R(
&
_output_shapes
:  : S

_output_shapes
: : T

_output_shapes
: : U

_output_shapes
: :,V(
&
_output_shapes
: @: W

_output_shapes
:@: X

_output_shapes
:@: Y

_output_shapes
:@:&Z"
 
_output_shapes
:
└А:![

_output_shapes	
:А:!\

_output_shapes	
:А:!]

_output_shapes	
:А:&^"
 
_output_shapes
:
АА:!_

_output_shapes	
:А:!`

_output_shapes	
:А:!a

_output_shapes	
:А:%b!

_output_shapes
:	А
: c

_output_shapes
:
:d

_output_shapes
: 
╠
Щ
)__inference_dense_10_layer_call_fn_350499

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_348453p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_350315

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╚
Ч
)__inference_dense_11_layer_call_fn_350626

inputs
unknown:	А

	unknown_0:

identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_348486o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347876

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
─
G
+__inference_dropout_17_layer_call_fn_350330

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_348399h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348049

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╟
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_348407

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ж

ў
C__inference_dense_9_layer_call_and_return_conditional_losses_350383

inputs2
matmul_readvariableop_resource:
└А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_13_layer_call_and_return_conditional_losses_348311

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
№	
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_350617

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╪Т
ЄB
"__inference__traced_restore_351264
file_prefix;
!assignvariableop_conv2d_12_kernel: /
!assignvariableop_1_conv2d_12_bias: =
/assignvariableop_2_batch_normalization_18_gamma: <
.assignvariableop_3_batch_normalization_18_beta: C
5assignvariableop_4_batch_normalization_18_moving_mean: G
9assignvariableop_5_batch_normalization_18_moving_variance: =
#assignvariableop_6_conv2d_13_kernel:  /
!assignvariableop_7_conv2d_13_bias: =
/assignvariableop_8_batch_normalization_19_gamma: <
.assignvariableop_9_batch_normalization_19_beta: D
6assignvariableop_10_batch_normalization_19_moving_mean: H
:assignvariableop_11_batch_normalization_19_moving_variance: >
$assignvariableop_12_conv2d_14_kernel:  0
"assignvariableop_13_conv2d_14_bias: >
0assignvariableop_14_batch_normalization_20_gamma: =
/assignvariableop_15_batch_normalization_20_beta: D
6assignvariableop_16_batch_normalization_20_moving_mean: H
:assignvariableop_17_batch_normalization_20_moving_variance: >
$assignvariableop_18_conv2d_15_kernel: @0
"assignvariableop_19_conv2d_15_bias:@>
0assignvariableop_20_batch_normalization_21_gamma:@=
/assignvariableop_21_batch_normalization_21_beta:@D
6assignvariableop_22_batch_normalization_21_moving_mean:@H
:assignvariableop_23_batch_normalization_21_moving_variance:@6
"assignvariableop_24_dense_9_kernel:
└А/
 assignvariableop_25_dense_9_bias:	А?
0assignvariableop_26_batch_normalization_22_gamma:	А>
/assignvariableop_27_batch_normalization_22_beta:	АE
6assignvariableop_28_batch_normalization_22_moving_mean:	АI
:assignvariableop_29_batch_normalization_22_moving_variance:	А7
#assignvariableop_30_dense_10_kernel:
АА0
!assignvariableop_31_dense_10_bias:	А?
0assignvariableop_32_batch_normalization_23_gamma:	А>
/assignvariableop_33_batch_normalization_23_beta:	АE
6assignvariableop_34_batch_normalization_23_moving_mean:	АI
:assignvariableop_35_batch_normalization_23_moving_variance:	А6
#assignvariableop_36_dense_11_kernel:	А
/
!assignvariableop_37_dense_11_bias:
)
assignvariableop_38_adamax_iter:	 +
!assignvariableop_39_adamax_beta_1: +
!assignvariableop_40_adamax_beta_2: *
 assignvariableop_41_adamax_decay: 2
(assignvariableop_42_adamax_learning_rate: #
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: G
-assignvariableop_47_adamax_conv2d_12_kernel_m: 9
+assignvariableop_48_adamax_conv2d_12_bias_m: G
9assignvariableop_49_adamax_batch_normalization_18_gamma_m: F
8assignvariableop_50_adamax_batch_normalization_18_beta_m: G
-assignvariableop_51_adamax_conv2d_13_kernel_m:  9
+assignvariableop_52_adamax_conv2d_13_bias_m: G
9assignvariableop_53_adamax_batch_normalization_19_gamma_m: F
8assignvariableop_54_adamax_batch_normalization_19_beta_m: G
-assignvariableop_55_adamax_conv2d_14_kernel_m:  9
+assignvariableop_56_adamax_conv2d_14_bias_m: G
9assignvariableop_57_adamax_batch_normalization_20_gamma_m: F
8assignvariableop_58_adamax_batch_normalization_20_beta_m: G
-assignvariableop_59_adamax_conv2d_15_kernel_m: @9
+assignvariableop_60_adamax_conv2d_15_bias_m:@G
9assignvariableop_61_adamax_batch_normalization_21_gamma_m:@F
8assignvariableop_62_adamax_batch_normalization_21_beta_m:@?
+assignvariableop_63_adamax_dense_9_kernel_m:
└А8
)assignvariableop_64_adamax_dense_9_bias_m:	АH
9assignvariableop_65_adamax_batch_normalization_22_gamma_m:	АG
8assignvariableop_66_adamax_batch_normalization_22_beta_m:	А@
,assignvariableop_67_adamax_dense_10_kernel_m:
АА9
*assignvariableop_68_adamax_dense_10_bias_m:	АH
9assignvariableop_69_adamax_batch_normalization_23_gamma_m:	АG
8assignvariableop_70_adamax_batch_normalization_23_beta_m:	А?
,assignvariableop_71_adamax_dense_11_kernel_m:	А
8
*assignvariableop_72_adamax_dense_11_bias_m:
G
-assignvariableop_73_adamax_conv2d_12_kernel_v: 9
+assignvariableop_74_adamax_conv2d_12_bias_v: G
9assignvariableop_75_adamax_batch_normalization_18_gamma_v: F
8assignvariableop_76_adamax_batch_normalization_18_beta_v: G
-assignvariableop_77_adamax_conv2d_13_kernel_v:  9
+assignvariableop_78_adamax_conv2d_13_bias_v: G
9assignvariableop_79_adamax_batch_normalization_19_gamma_v: F
8assignvariableop_80_adamax_batch_normalization_19_beta_v: G
-assignvariableop_81_adamax_conv2d_14_kernel_v:  9
+assignvariableop_82_adamax_conv2d_14_bias_v: G
9assignvariableop_83_adamax_batch_normalization_20_gamma_v: F
8assignvariableop_84_adamax_batch_normalization_20_beta_v: G
-assignvariableop_85_adamax_conv2d_15_kernel_v: @9
+assignvariableop_86_adamax_conv2d_15_bias_v:@G
9assignvariableop_87_adamax_batch_normalization_21_gamma_v:@F
8assignvariableop_88_adamax_batch_normalization_21_beta_v:@?
+assignvariableop_89_adamax_dense_9_kernel_v:
└А8
)assignvariableop_90_adamax_dense_9_bias_v:	АH
9assignvariableop_91_adamax_batch_normalization_22_gamma_v:	АG
8assignvariableop_92_adamax_batch_normalization_22_beta_v:	А@
,assignvariableop_93_adamax_dense_10_kernel_v:
АА9
*assignvariableop_94_adamax_dense_10_bias_v:	АH
9assignvariableop_95_adamax_batch_normalization_23_gamma_v:	АG
8assignvariableop_96_adamax_batch_normalization_23_beta_v:	А?
,assignvariableop_97_adamax_dense_11_kernel_v:	А
8
*assignvariableop_98_adamax_dense_11_bias_v:

identity_100ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98▓7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*╪6
value╬6B╦6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╗
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*▌
value╙B╨dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Х
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ж
_output_shapesУ
Р::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_18_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_18_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_18_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_18_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_19_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_19_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_19_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_19_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_20_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_20_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_20_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_20_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_21_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_21_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_21_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_21_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_9_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_9_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_22_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_22_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_22_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_22_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_10_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_10_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_23_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_23_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_23_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_23_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_11_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_11_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:Р
AssignVariableOp_38AssignVariableOpassignvariableop_38_adamax_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_39AssignVariableOp!assignvariableop_39_adamax_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_40AssignVariableOp!assignvariableop_40_adamax_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_41AssignVariableOp assignvariableop_41_adamax_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adamax_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adamax_conv2d_12_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adamax_conv2d_12_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_49AssignVariableOp9assignvariableop_49_adamax_batch_normalization_18_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adamax_batch_normalization_18_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adamax_conv2d_13_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adamax_conv2d_13_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_53AssignVariableOp9assignvariableop_53_adamax_batch_normalization_19_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_54AssignVariableOp8assignvariableop_54_adamax_batch_normalization_19_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adamax_conv2d_14_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adamax_conv2d_14_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_57AssignVariableOp9assignvariableop_57_adamax_batch_normalization_20_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_58AssignVariableOp8assignvariableop_58_adamax_batch_normalization_20_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_59AssignVariableOp-assignvariableop_59_adamax_conv2d_15_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_60AssignVariableOp+assignvariableop_60_adamax_conv2d_15_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_61AssignVariableOp9assignvariableop_61_adamax_batch_normalization_21_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_62AssignVariableOp8assignvariableop_62_adamax_batch_normalization_21_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adamax_dense_9_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adamax_dense_9_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_65AssignVariableOp9assignvariableop_65_adamax_batch_normalization_22_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_66AssignVariableOp8assignvariableop_66_adamax_batch_normalization_22_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adamax_dense_10_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adamax_dense_10_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_69AssignVariableOp9assignvariableop_69_adamax_batch_normalization_23_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_70AssignVariableOp8assignvariableop_70_adamax_batch_normalization_23_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adamax_dense_11_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adamax_dense_11_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_73AssignVariableOp-assignvariableop_73_adamax_conv2d_12_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adamax_conv2d_12_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_75AssignVariableOp9assignvariableop_75_adamax_batch_normalization_18_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_76AssignVariableOp8assignvariableop_76_adamax_batch_normalization_18_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_77AssignVariableOp-assignvariableop_77_adamax_conv2d_13_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_78AssignVariableOp+assignvariableop_78_adamax_conv2d_13_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_79AssignVariableOp9assignvariableop_79_adamax_batch_normalization_19_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_80AssignVariableOp8assignvariableop_80_adamax_batch_normalization_19_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_81AssignVariableOp-assignvariableop_81_adamax_conv2d_14_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_82AssignVariableOp+assignvariableop_82_adamax_conv2d_14_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_83AssignVariableOp9assignvariableop_83_adamax_batch_normalization_20_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_84AssignVariableOp8assignvariableop_84_adamax_batch_normalization_20_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_85AssignVariableOp-assignvariableop_85_adamax_conv2d_15_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_86AssignVariableOp+assignvariableop_86_adamax_conv2d_15_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_87AssignVariableOp9assignvariableop_87_adamax_batch_normalization_21_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_88AssignVariableOp8assignvariableop_88_adamax_batch_normalization_21_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adamax_dense_9_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adamax_dense_9_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_91AssignVariableOp9assignvariableop_91_adamax_batch_normalization_22_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_92AssignVariableOp8assignvariableop_92_adamax_batch_normalization_22_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adamax_dense_10_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adamax_dense_10_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_95AssignVariableOp9assignvariableop_95_adamax_batch_normalization_23_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_96AssignVariableOp8assignvariableop_96_adamax_batch_normalization_23_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_97AssignVariableOp,assignvariableop_97_adamax_dense_11_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_98AssignVariableOp*assignvariableop_98_adamax_dense_11_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╤
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: X
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: ╛
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 "%
identity_100Identity_100:output:0*▌
_input_shapes╦
╚: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
═
Э
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347909

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ф	
╥
7__inference_batch_normalization_19_layer_call_fn_350051

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347940Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ф	
╥
7__inference_batch_normalization_18_layer_call_fn_349969

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347876Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░%
я
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348256

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	АИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ам
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Аъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_350097

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╗
L
0__inference_max_pooling2d_6_layer_call_fn_350092

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_347960Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я
Я
*__inference_conv2d_12_layer_call_fn_349932

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_348285w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ц
d
+__inference_dropout_15_layer_call_fn_350107

inputs
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_348740w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_12_layer_call_and_return_conditional_losses_348285

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_350069

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░%
я
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_350463

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	АИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ам
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Аъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
№h
╔
H__inference_sequential_3_layer_call_and_return_conditional_losses_349208
conv2d_12_input*
conv2d_12_349110: 
conv2d_12_349112: +
batch_normalization_18_349115: +
batch_normalization_18_349117: +
batch_normalization_18_349119: +
batch_normalization_18_349121: *
conv2d_13_349124:  
conv2d_13_349126: +
batch_normalization_19_349129: +
batch_normalization_19_349131: +
batch_normalization_19_349133: +
batch_normalization_19_349135: *
conv2d_14_349140:  
conv2d_14_349142: +
batch_normalization_20_349145: +
batch_normalization_20_349147: +
batch_normalization_20_349149: +
batch_normalization_20_349151: *
conv2d_15_349155: @
conv2d_15_349157:@+
batch_normalization_21_349160:@+
batch_normalization_21_349162:@+
batch_normalization_21_349164:@+
batch_normalization_21_349166:@"
dense_9_349172:
└А
dense_9_349174:	А,
batch_normalization_22_349177:	А,
batch_normalization_22_349179:	А,
batch_normalization_22_349181:	А,
batch_normalization_22_349183:	А#
dense_10_349187:
АА
dense_10_349189:	А,
batch_normalization_23_349192:	А,
batch_normalization_23_349194:	А,
batch_normalization_23_349196:	А,
batch_normalization_23_349198:	А"
dense_11_349202:	А

dense_11_349204:

identityИв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_12/StatefulPartitionedCallв!conv2d_13/StatefulPartitionedCallв!conv2d_14/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв dense_10/StatefulPartitionedCallв dense_11/StatefulPartitionedCallвdense_9/StatefulPartitionedCallИ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_349110conv2d_12_349112*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_348285Щ
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_18_349115batch_normalization_18_349117batch_normalization_18_349119batch_normalization_18_349121*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347845░
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_13_349124conv2d_13_349126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_348311Щ
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_19_349129batch_normalization_19_349131batch_normalization_19_349133batch_normalization_19_349135*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347909В
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_347960щ
dropout_15/PartitionedCallPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_348332Ь
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0conv2d_14_349140conv2d_14_349142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_348345Щ
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_20_349145batch_normalization_20_349147batch_normalization_20_349149batch_normalization_20_349151*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_347985°
dropout_16/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_348365Ь
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0conv2d_15_349155conv2d_15_349157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_348378Щ
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_21_349160batch_normalization_21_349162batch_normalization_21_349164batch_normalization_21_349166*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348049В
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_348100щ
dropout_17/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_348399█
flatten_3/PartitionedCallPartitionedCall#dropout_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_348407М
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_349172dense_9_349174*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_348420Р
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_22_349177batch_normalization_22_349179batch_normalization_22_349181batch_normalization_22_349183*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348127ё
dropout_18/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_348440С
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_10_349187dense_10_349189*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_348453С
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_23_349192batch_normalization_23_349194batch_normalization_23_349196batch_normalization_23_349198*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348209ё
dropout_19/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_348473Р
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_11_349202dense_11_349204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_348486x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ф
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_12_input
ж

ў
C__inference_dense_9_layer_call_and_return_conditional_losses_348420

inputs2
matmul_readvariableop_resource:
└А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
┤

e
F__inference_dropout_15_layer_call_and_return_conditional_losses_348740

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┤

e
F__inference_dropout_15_layer_call_and_return_conditional_losses_350124

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348080

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╗
L
0__inference_max_pooling2d_7_layer_call_fn_350320

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_348100Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ц
d
+__inference_dropout_17_layer_call_fn_350335

inputs
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_348674w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
▌
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_350478

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┤

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_348707

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_350297

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_348100

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
░%
я
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348174

inputs6
'assignmovingavg_readvariableop_resource:	А8
)assignmovingavg_1_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А0
!batchnorm_readvariableop_resource:	А
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	АИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Аy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ам
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:А┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Аw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Аъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
№	
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_348635

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
с
╡
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_350556

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·
d
+__inference_dropout_18_layer_call_fn_350473

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_348635p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
з

°
D__inference_dense_10_layer_call_and_return_conditional_losses_348453

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
∙
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_348332

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:          c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╩
Ш
(__inference_dense_9_layer_call_fn_350372

inputs
unknown:
└А
	unknown_0:	А
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_348420p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_14_layer_call_and_return_conditional_losses_350144

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╡
╓
7__inference_batch_normalization_23_layer_call_fn_350523

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348209p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┤╕
┤&
H__inference_sequential_3_layer_call_and_return_conditional_losses_349840

inputsB
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: <
.batch_normalization_18_readvariableop_resource: >
0batch_normalization_18_readvariableop_1_resource: M
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_13_conv2d_readvariableop_resource:  7
)conv2d_13_biasadd_readvariableop_resource: <
.batch_normalization_19_readvariableop_resource: >
0batch_normalization_19_readvariableop_1_resource: M
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_14_conv2d_readvariableop_resource:  7
)conv2d_14_biasadd_readvariableop_resource: <
.batch_normalization_20_readvariableop_resource: >
0batch_normalization_20_readvariableop_1_resource: M
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_15_conv2d_readvariableop_resource: @7
)conv2d_15_biasadd_readvariableop_resource:@<
.batch_normalization_21_readvariableop_resource:@>
0batch_normalization_21_readvariableop_1_resource:@M
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:@:
&dense_9_matmul_readvariableop_resource:
└А6
'dense_9_biasadd_readvariableop_resource:	АM
>batch_normalization_22_assignmovingavg_readvariableop_resource:	АO
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:	АK
<batch_normalization_22_batchnorm_mul_readvariableop_resource:	АG
8batch_normalization_22_batchnorm_readvariableop_resource:	А;
'dense_10_matmul_readvariableop_resource:
АА7
(dense_10_biasadd_readvariableop_resource:	АM
>batch_normalization_23_assignmovingavg_readvariableop_resource:	АO
@batch_normalization_23_assignmovingavg_1_readvariableop_resource:	АK
<batch_normalization_23_batchnorm_mul_readvariableop_resource:	АG
8batch_normalization_23_batchnorm_readvariableop_resource:	А:
'dense_11_matmul_readvariableop_resource:	А
6
(dense_11_biasadd_readvariableop_resource:

identityИв%batch_normalization_18/AssignNewValueв'batch_normalization_18/AssignNewValue_1в6batch_normalization_18/FusedBatchNormV3/ReadVariableOpв8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_18/ReadVariableOpв'batch_normalization_18/ReadVariableOp_1в%batch_normalization_19/AssignNewValueв'batch_normalization_19/AssignNewValue_1в6batch_normalization_19/FusedBatchNormV3/ReadVariableOpв8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_19/ReadVariableOpв'batch_normalization_19/ReadVariableOp_1в%batch_normalization_20/AssignNewValueв'batch_normalization_20/AssignNewValue_1в6batch_normalization_20/FusedBatchNormV3/ReadVariableOpв8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_20/ReadVariableOpв'batch_normalization_20/ReadVariableOp_1в%batch_normalization_21/AssignNewValueв'batch_normalization_21/AssignNewValue_1в6batch_normalization_21/FusedBatchNormV3/ReadVariableOpв8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_21/ReadVariableOpв'batch_normalization_21/ReadVariableOp_1в&batch_normalization_22/AssignMovingAvgв5batch_normalization_22/AssignMovingAvg/ReadVariableOpв(batch_normalization_22/AssignMovingAvg_1в7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_22/batchnorm/ReadVariableOpв3batch_normalization_22/batchnorm/mul/ReadVariableOpв&batch_normalization_23/AssignMovingAvgв5batch_normalization_23/AssignMovingAvg/ReadVariableOpв(batch_normalization_23/AssignMovingAvg_1в7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_23/batchnorm/ReadVariableOpв3batch_normalization_23/batchnorm/mul/ReadVariableOpв conv2d_12/BiasAdd/ReadVariableOpвconv2d_12/Conv2D/ReadVariableOpв conv2d_13/BiasAdd/ReadVariableOpвconv2d_13/Conv2D/ReadVariableOpв conv2d_14/BiasAdd/ReadVariableOpвconv2d_14/Conv2D/ReadVariableOpв conv2d_15/BiasAdd/ReadVariableOpвconv2d_15/Conv2D/ReadVariableOpвdense_10/BiasAdd/ReadVariableOpвdense_10/MatMul/ReadVariableOpвdense_11/BiasAdd/ReadVariableOpвdense_11/MatMul/ReadVariableOpвdense_9/BiasAdd/ReadVariableOpвdense_9/MatMul/ReadVariableOpР
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0н
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ж
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          l
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:          Р
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╢
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0═
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<М
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ц
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Р
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0╥
conv2d_13/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ж
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:          Р
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╢
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0═
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<М
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ц
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0╝
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_19/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Ь
dropout_15/dropout/MulMul max_pooling2d_6/MaxPool:output:0!dropout_15/dropout/Const:output:0*
T0*/
_output_shapes
:          h
dropout_15/dropout/ShapeShape max_pooling2d_6/MaxPool:output:0*
T0*
_output_shapes
:к
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype0f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╧
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          Н
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          Т
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*/
_output_shapes
:          Р
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0├
conv2d_14/Conv2DConv2Ddropout_15/dropout/Mul_1:z:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
Ж
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:          Р
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0╢
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0═
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<М
%batch_normalization_20/AssignNewValueAssignVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource4batch_normalization_20/FusedBatchNormV3:batch_mean:07^batch_normalization_20/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ц
'batch_normalization_20/AssignNewValue_1AssignVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_20/FusedBatchNormV3:batch_variance:09^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?з
dropout_16/dropout/MulMul+batch_normalization_20/FusedBatchNormV3:y:0!dropout_16/dropout/Const:output:0*
T0*/
_output_shapes
:          s
dropout_16/dropout/ShapeShape+batch_normalization_20/FusedBatchNormV3:y:0*
T0*
_output_shapes
:к
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╧
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          Н
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          Т
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*/
_output_shapes
:          Р
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0├
conv2d_15/Conv2DConv2Ddropout_16/dropout/Mul_1:z:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         @Р
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0═
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<М
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ц
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0╝
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Ь
dropout_17/dropout/MulMul max_pooling2d_7/MaxPool:output:0!dropout_17/dropout/Const:output:0*
T0*/
_output_shapes
:         @h
dropout_17/dropout/ShapeShape max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:к
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╧
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @Н
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @Т
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*/
_output_shapes
:         @`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  З
flatten_3/ReshapeReshapedropout_17/dropout/Mul_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:         └Ж
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0О
dense_9/MatMulMatMulflatten_3/Reshape:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*(
_output_shapes
:         А
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ┬
#batch_normalization_22/moments/meanMeandense_9/Relu:activations:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(У
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:	А╩
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_9/Relu:activations:04batch_normalization_22/moments/StopGradient:output:0*
T0*(
_output_shapes
:         АГ
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ф
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(Ь
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 в
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 q
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<▒
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0╟
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes	
:А╛
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:05batch_normalization_22/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:АИ
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<╡
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0═
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А─
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:07batch_normalization_22/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:АР
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╖
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes	
:Ан
3batch_normalization_22/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_22_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0║
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:0;batch_normalization_22/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аж
&batch_normalization_22/batchnorm/mul_1Muldense_9/Relu:activations:0(batch_normalization_22/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ао
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ае
/batch_normalization_22/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_22_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
$batch_normalization_22/batchnorm/subSub7batch_normalization_22/batchnorm/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А╕
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А]
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Я
dropout_18/dropout/MulMul*batch_normalization_22/batchnorm/add_1:z:0!dropout_18/dropout/Const:output:0*
T0*(
_output_shapes
:         Аr
dropout_18/dropout/ShapeShape*batch_normalization_22/batchnorm/add_1:z:0*
T0*
_output_shapes
:г
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АЖ
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         АЛ
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*(
_output_shapes
:         АИ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
dense_10/MatMulMatMuldropout_18/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:         А
5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ├
#batch_normalization_23/moments/meanMeandense_10/Relu:activations:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(У
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes
:	А╦
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:04batch_normalization_23/moments/StopGradient:output:0*
T0*(
_output_shapes
:         АГ
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ф
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(Ь
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 в
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 q
,batch_normalization_23/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<▒
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_23_assignmovingavg_readvariableop_resource*
_output_shapes	
:А*
dtype0╟
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0*
T0*
_output_shapes	
:А╛
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:АИ
&batch_normalization_23/AssignMovingAvgAssignSubVariableOp>batch_normalization_23_assignmovingavg_readvariableop_resource.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_23/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<╡
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_23_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:А*
dtype0═
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:А─
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:АР
(batch_normalization_23/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_23_assignmovingavg_1_readvariableop_resource0batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╖
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:Ан
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0║
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аз
&batch_normalization_23/batchnorm/mul_1Muldense_10/Relu:activations:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ао
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ае
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А╕
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А]
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Я
dropout_19/dropout/MulMul*batch_normalization_23/batchnorm/add_1:z:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:         Аr
dropout_19/dropout/ShapeShape*batch_normalization_23/batchnorm/add_1:z:0*
T0*
_output_shapes
:г
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АЖ
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         АЛ
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:         АЗ
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0С
dense_11/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Д
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0С
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_11/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
▌
NoOpNoOp&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1&^batch_normalization_20/AssignNewValue(^batch_normalization_20/AssignNewValue_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_22/batchnorm/ReadVariableOp4^batch_normalization_22/batchnorm/mul/ReadVariableOp'^batch_normalization_23/AssignMovingAvg6^batch_normalization_23/AssignMovingAvg/ReadVariableOp)^batch_normalization_23/AssignMovingAvg_18^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp4^batch_normalization_23/batchnorm/mul/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12N
%batch_normalization_20/AssignNewValue%batch_normalization_20/AssignNewValue2R
'batch_normalization_20/AssignNewValue_1'batch_normalization_20/AssignNewValue_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_22/batchnorm/ReadVariableOp/batch_normalization_22/batchnorm/ReadVariableOp2j
3batch_normalization_22/batchnorm/mul/ReadVariableOp3batch_normalization_22/batchnorm/mul/ReadVariableOp2P
&batch_normalization_23/AssignMovingAvg&batch_normalization_23/AssignMovingAvg2n
5batch_normalization_23/AssignMovingAvg/ReadVariableOp5batch_normalization_23/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_23/AssignMovingAvg_1(batch_normalization_23/AssignMovingAvg_12r
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┤

e
F__inference_dropout_17_layer_call_and_return_conditional_losses_348674

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_18_layer_call_fn_349956

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347845Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_347985

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347940

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Й
Й	
-__inference_sequential_3_layer_call_fn_349107
conv2d_12_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
└А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А


unknown_36:

identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_348947o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_12_input
Ф	
╥
7__inference_batch_normalization_20_layer_call_fn_350170

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_348016Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
№	
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_350490

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_350188

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
я
Я
*__inference_conv2d_15_layer_call_fn_350242

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_348378w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
▌
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_348440

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_350325

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_12_layer_call_and_return_conditional_losses_349943

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
·
А	
-__inference_sequential_3_layer_call_fn_349396

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
└А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А


unknown_36:

identityИвStatefulPartitionedCall╧
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_348493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
№	
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_348602

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_350206

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ц	
╥
7__inference_batch_normalization_20_layer_call_fn_350157

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_347985Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
∙
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_350340

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
·
d
+__inference_dropout_19_layer_call_fn_350600

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_348602p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_350087

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_15_layer_call_and_return_conditional_losses_348378

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┤

e
F__inference_dropout_16_layer_call_and_return_conditional_losses_350233

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┤

e
F__inference_dropout_17_layer_call_and_return_conditional_losses_350352

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>о
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
с
╡
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_350429

inputs0
!batchnorm_readvariableop_resource:	А4
%batchnorm_mul_readvariableop_resource:	А2
#batchnorm_readvariableop_1_resource:	А2
#batchnorm_readvariableop_2_resource:	А
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:АQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         А║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ю
А	
-__inference_sequential_3_layer_call_fn_349477

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
└А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А


unknown_36:

identityИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
 #$%&*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_348947o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Х
Й	
-__inference_sequential_3_layer_call_fn_348572
conv2d_12_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
└А

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А

unknown_29:
АА

unknown_30:	А

unknown_31:	А

unknown_32:	А

unknown_33:	А

unknown_34:	А

unknown_35:	А


unknown_36:

identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_348493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_12_input
═
Э
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347845

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_347960

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
█
┴
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_350005

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<░
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0║
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Д
■
E__inference_conv2d_13_layer_call_and_return_conditional_losses_350025

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
и
G
+__inference_dropout_19_layer_call_fn_350595

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_348473a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
д

Ў
D__inference_dense_11_layer_call_and_return_conditional_losses_350637

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
рp
В
H__inference_sequential_3_layer_call_and_return_conditional_losses_349309
conv2d_12_input*
conv2d_12_349211: 
conv2d_12_349213: +
batch_normalization_18_349216: +
batch_normalization_18_349218: +
batch_normalization_18_349220: +
batch_normalization_18_349222: *
conv2d_13_349225:  
conv2d_13_349227: +
batch_normalization_19_349230: +
batch_normalization_19_349232: +
batch_normalization_19_349234: +
batch_normalization_19_349236: *
conv2d_14_349241:  
conv2d_14_349243: +
batch_normalization_20_349246: +
batch_normalization_20_349248: +
batch_normalization_20_349250: +
batch_normalization_20_349252: *
conv2d_15_349256: @
conv2d_15_349258:@+
batch_normalization_21_349261:@+
batch_normalization_21_349263:@+
batch_normalization_21_349265:@+
batch_normalization_21_349267:@"
dense_9_349273:
└А
dense_9_349275:	А,
batch_normalization_22_349278:	А,
batch_normalization_22_349280:	А,
batch_normalization_22_349282:	А,
batch_normalization_22_349284:	А#
dense_10_349288:
АА
dense_10_349290:	А,
batch_normalization_23_349293:	А,
batch_normalization_23_349295:	А,
batch_normalization_23_349297:	А,
batch_normalization_23_349299:	А"
dense_11_349303:	А

dense_11_349305:

identityИв.batch_normalization_18/StatefulPartitionedCallв.batch_normalization_19/StatefulPartitionedCallв.batch_normalization_20/StatefulPartitionedCallв.batch_normalization_21/StatefulPartitionedCallв.batch_normalization_22/StatefulPartitionedCallв.batch_normalization_23/StatefulPartitionedCallв!conv2d_12/StatefulPartitionedCallв!conv2d_13/StatefulPartitionedCallв!conv2d_14/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв dense_10/StatefulPartitionedCallв dense_11/StatefulPartitionedCallвdense_9/StatefulPartitionedCallв"dropout_15/StatefulPartitionedCallв"dropout_16/StatefulPartitionedCallв"dropout_17/StatefulPartitionedCallв"dropout_18/StatefulPartitionedCallв"dropout_19/StatefulPartitionedCallИ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_349211conv2d_12_349213*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_348285Ч
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_18_349216batch_normalization_18_349218batch_normalization_18_349220batch_normalization_18_349222*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_347876░
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_13_349225conv2d_13_349227*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_348311Ч
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_19_349230batch_normalization_19_349232batch_normalization_19_349234batch_normalization_19_349236*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_347940В
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_347960∙
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_348740д
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0conv2d_14_349241conv2d_14_349243*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_348345Ч
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_20_349246batch_normalization_20_349248batch_normalization_20_349250batch_normalization_20_349252*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_348016н
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_348707д
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0conv2d_15_349256conv2d_15_349258*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_348378Ч
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_21_349261batch_normalization_21_349263batch_normalization_21_349265batch_normalization_21_349267*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_348080В
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_348100Ю
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_348674у
flatten_3/PartitionedCallPartitionedCall+dropout_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_348407М
dense_9/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_9_349273dense_9_349275*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_348420О
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0batch_normalization_22_349278batch_normalization_22_349280batch_normalization_22_349282batch_normalization_22_349284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_348174ж
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_348635Щ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_10_349288dense_10_349290*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_348453П
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_23_349293batch_normalization_23_349295batch_normalization_23_349297batch_normalization_23_349299*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_348256ж
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_348602Ш
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_11_349303dense_11_349305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_348486x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
Э
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall:` \
/
_output_shapes
:         
)
_user_specified_nameconv2d_12_input
▌
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_348473

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
д

Ў
D__inference_dense_11_layer_call_and_return_conditional_losses_348486

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*├
serving_defaultп
S
conv2d_12_input@
!serving_default_conv2d_12_input:0         <
dense_110
StatefulPartitionedCall:0         
tensorflow/serving/predict:ЬФ
╫
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
╗

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
'axis
	(gamma
)beta
*moving_mean
+moving_variance
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
е
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O_random_generator
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
╝
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i_random_generator
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
taxis
	ugamma
vbeta
wmoving_mean
xmoving_variance
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
к
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й_random_generator
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
л
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Тkernel
	Уbias
Ф	variables
Хtrainable_variables
Цregularization_losses
Ч	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
	Ъaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
├
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й_random_generator
к__call__
+л&call_and_return_all_conditional_losses"
_tf_keras_layer
├
мkernel
	нbias
о	variables
пtrainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
	┤axis

╡gamma
	╢beta
╖moving_mean
╕moving_variance
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"
_tf_keras_layer
├
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
├_random_generator
─__call__
+┼&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╞kernel
	╟bias
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
╠__call__
+═&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
	╬iter
╧beta_1
╨beta_2

╤decay
╥learning_ratem═ m╬(m╧)m╨2m╤3m╥;m╙<m╘Rm╒Sm╓[m╫\m╪lm┘mm┌um█vm▄	Тm▌	Уm▐	Ыm▀	Ьmр	мmс	нmт	╡mу	╢mф	╞mх	╟mцvч vш(vщ)vъ2vы3vь;vэ<vюRvяSvЁ[vё\vЄlvєmvЇuvїvvЎ	Тvў	Уv°	Ыv∙	Ьv·	мv√	нv№	╡v¤	╢v■	╞v 	╟vА"
	optimizer
╘
0
 1
(2
)3
*4
+5
26
37
;8
<9
=10
>11
R12
S13
[14
\15
]16
^17
l18
m19
u20
v21
w22
x23
Т24
У25
Ы26
Ь27
Э28
Ю29
м30
н31
╡32
╢33
╖34
╕35
╞36
╟37"
trackable_list_wrapper
Ё
0
 1
(2
)3
24
35
;6
<7
R8
S9
[10
\11
l12
m13
u14
v15
Т16
У17
Ы18
Ь19
м20
н21
╡22
╢23
╞24
╟25"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
В2 
-__inference_sequential_3_layer_call_fn_348572
-__inference_sequential_3_layer_call_fn_349396
-__inference_sequential_3_layer_call_fn_349477
-__inference_sequential_3_layer_call_fn_349107└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_349627
H__inference_sequential_3_layer_call_and_return_conditional_losses_349840
H__inference_sequential_3_layer_call_and_return_conditional_losses_349208
H__inference_sequential_3_layer_call_and_return_conditional_losses_349309└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘B╤
!__inference__wrapped_model_347823conv2d_12_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-
╪serving_default"
signature_map
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
▌layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
╘2╤
*__inference_conv2d_12_layer_call_fn_349932в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_12_layer_call_and_return_conditional_losses_349943в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
*:( 2batch_normalization_18/gamma
):' 2batch_normalization_18/beta
2:0  (2"batch_normalization_18/moving_mean
6:4  (2&batch_normalization_18/moving_variance
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▐non_trainable_variables
▀layers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_18_layer_call_fn_349956
7__inference_batch_normalization_18_layer_call_fn_349969┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_349987
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_350005┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
*:(  2conv2d_13/kernel
: 2conv2d_13/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
╘2╤
*__inference_conv2d_13_layer_call_fn_350014в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_13_layer_call_and_return_conditional_losses_350025в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
*:( 2batch_normalization_19/gamma
):' 2batch_normalization_19/beta
2:0  (2"batch_normalization_19/moving_mean
6:4  (2&batch_normalization_19/moving_variance
<
;0
<1
=2
>3"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_19_layer_call_fn_350038
7__inference_batch_normalization_19_layer_call_fn_350051┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_350069
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_350087┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
эnon_trainable_variables
юlayers
яmetrics
 Ёlayer_regularization_losses
ёlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
┌2╫
0__inference_max_pooling2d_6_layer_call_fn_350092в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_350097в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_15_layer_call_fn_350102
+__inference_dropout_15_layer_call_fn_350107┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_15_layer_call_and_return_conditional_losses_350112
F__inference_dropout_15_layer_call_and_return_conditional_losses_350124┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
*:(  2conv2d_14/kernel
: 2conv2d_14/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
╘2╤
*__inference_conv2d_14_layer_call_fn_350133в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_14_layer_call_and_return_conditional_losses_350144в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
*:( 2batch_normalization_20/gamma
):' 2batch_normalization_20/beta
2:0  (2"batch_normalization_20/moving_mean
6:4  (2&batch_normalization_20/moving_variance
<
[0
\1
]2
^3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
№non_trainable_variables
¤layers
■metrics
  layer_regularization_losses
Аlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_20_layer_call_fn_350157
7__inference_batch_normalization_20_layer_call_fn_350170┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_350188
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_350206┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_16_layer_call_fn_350211
+__inference_dropout_16_layer_call_fn_350216┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_16_layer_call_and_return_conditional_losses_350221
F__inference_dropout_16_layer_call_and_return_conditional_losses_350233┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
*:( @2conv2d_15/kernel
:@2conv2d_15/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
╘2╤
*__inference_conv2d_15_layer_call_fn_350242в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv2d_15_layer_call_and_return_conditional_losses_350253в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
*:(@2batch_normalization_21/gamma
):'@2batch_normalization_21/beta
2:0@ (2"batch_normalization_21/moving_mean
6:4@ (2&batch_normalization_21/moving_variance
<
u0
v1
w2
x3"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_21_layer_call_fn_350266
7__inference_batch_normalization_21_layer_call_fn_350279┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_350297
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_350315┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╖
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
┌2╫
0__inference_max_pooling2d_7_layer_call_fn_350320в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_350325в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_17_layer_call_fn_350330
+__inference_dropout_17_layer_call_fn_350335┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_17_layer_call_and_return_conditional_losses_350340
F__inference_dropout_17_layer_call_and_return_conditional_losses_350352┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
╘2╤
*__inference_flatten_3_layer_call_fn_350357в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_flatten_3_layer_call_and_return_conditional_losses_350363в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
": 
└А2dense_9/kernel
:А2dense_9/bias
0
Т0
У1"
trackable_list_wrapper
0
Т0
У1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
Ф	variables
Хtrainable_variables
Цregularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
╥2╧
(__inference_dense_9_layer_call_fn_350372в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_9_layer_call_and_return_conditional_losses_350383в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
+:)А2batch_normalization_22/gamma
*:(А2batch_normalization_22/beta
3:1А (2"batch_normalization_22/moving_mean
7:5А (2&batch_normalization_22/moving_variance
@
Ы0
Ь1
Э2
Ю3"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_22_layer_call_fn_350396
7__inference_batch_normalization_22_layer_call_fn_350409┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_350429
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_350463┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_18_layer_call_fn_350468
+__inference_dropout_18_layer_call_fn_350473┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_18_layer_call_and_return_conditional_losses_350478
F__inference_dropout_18_layer_call_and_return_conditional_losses_350490┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
#:!
АА2dense_10/kernel
:А2dense_10/bias
0
м0
н1"
trackable_list_wrapper
0
м0
н1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
оnon_trainable_variables
пlayers
░metrics
 ▒layer_regularization_losses
▓layer_metrics
о	variables
пtrainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_dense_10_layer_call_fn_350499в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_10_layer_call_and_return_conditional_losses_350510в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
+:)А2batch_normalization_23/gamma
*:(А2batch_normalization_23/beta
3:1А (2"batch_normalization_23/moving_mean
7:5А (2&batch_normalization_23/moving_variance
@
╡0
╢1
╖2
╕3"
trackable_list_wrapper
0
╡0
╢1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
╣	variables
║trainable_variables
╗regularization_losses
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
м2й
7__inference_batch_normalization_23_layer_call_fn_350523
7__inference_batch_normalization_23_layer_call_fn_350536┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_350556
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_350590┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
┐	variables
└trainable_variables
┴regularization_losses
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ф2С
+__inference_dropout_19_layer_call_fn_350595
+__inference_dropout_19_layer_call_fn_350600┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout_19_layer_call_and_return_conditional_losses_350605
F__inference_dropout_19_layer_call_and_return_conditional_losses_350617┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
": 	А
2dense_11/kernel
:
2dense_11/bias
0
╞0
╟1"
trackable_list_wrapper
0
╞0
╟1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
╚	variables
╔trainable_variables
╩regularization_losses
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_dense_11_layer_call_fn_350626в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_dense_11_layer_call_and_return_conditional_losses_350637в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2Adamax/iter
: (2Adamax/beta_1
: (2Adamax/beta_2
: (2Adamax/decay
: (2Adamax/learning_rate
z
*0
+1
=2
>3
]4
^5
w6
x7
Э8
Ю9
╖10
╕11"
trackable_list_wrapper
╛
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
0
┬0
├1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
$__inference_signature_wrapper_349923conv2d_12_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
╖0
╕1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

─total

┼count
╞	variables
╟	keras_api"
_tf_keras_metric
c

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
─0
┼1"
trackable_list_wrapper
.
╞	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╚0
╔1"
trackable_list_wrapper
.
╦	variables"
_generic_user_object
1:/ 2Adamax/conv2d_12/kernel/m
#:! 2Adamax/conv2d_12/bias/m
1:/ 2%Adamax/batch_normalization_18/gamma/m
0:. 2$Adamax/batch_normalization_18/beta/m
1:/  2Adamax/conv2d_13/kernel/m
#:! 2Adamax/conv2d_13/bias/m
1:/ 2%Adamax/batch_normalization_19/gamma/m
0:. 2$Adamax/batch_normalization_19/beta/m
1:/  2Adamax/conv2d_14/kernel/m
#:! 2Adamax/conv2d_14/bias/m
1:/ 2%Adamax/batch_normalization_20/gamma/m
0:. 2$Adamax/batch_normalization_20/beta/m
1:/ @2Adamax/conv2d_15/kernel/m
#:!@2Adamax/conv2d_15/bias/m
1:/@2%Adamax/batch_normalization_21/gamma/m
0:.@2$Adamax/batch_normalization_21/beta/m
):'
└А2Adamax/dense_9/kernel/m
": А2Adamax/dense_9/bias/m
2:0А2%Adamax/batch_normalization_22/gamma/m
1:/А2$Adamax/batch_normalization_22/beta/m
*:(
АА2Adamax/dense_10/kernel/m
#:!А2Adamax/dense_10/bias/m
2:0А2%Adamax/batch_normalization_23/gamma/m
1:/А2$Adamax/batch_normalization_23/beta/m
):'	А
2Adamax/dense_11/kernel/m
": 
2Adamax/dense_11/bias/m
1:/ 2Adamax/conv2d_12/kernel/v
#:! 2Adamax/conv2d_12/bias/v
1:/ 2%Adamax/batch_normalization_18/gamma/v
0:. 2$Adamax/batch_normalization_18/beta/v
1:/  2Adamax/conv2d_13/kernel/v
#:! 2Adamax/conv2d_13/bias/v
1:/ 2%Adamax/batch_normalization_19/gamma/v
0:. 2$Adamax/batch_normalization_19/beta/v
1:/  2Adamax/conv2d_14/kernel/v
#:! 2Adamax/conv2d_14/bias/v
1:/ 2%Adamax/batch_normalization_20/gamma/v
0:. 2$Adamax/batch_normalization_20/beta/v
1:/ @2Adamax/conv2d_15/kernel/v
#:!@2Adamax/conv2d_15/bias/v
1:/@2%Adamax/batch_normalization_21/gamma/v
0:.@2$Adamax/batch_normalization_21/beta/v
):'
└А2Adamax/dense_9/kernel/v
": А2Adamax/dense_9/bias/v
2:0А2%Adamax/batch_normalization_22/gamma/v
1:/А2$Adamax/batch_normalization_22/beta/v
*:(
АА2Adamax/dense_10/kernel/v
#:!А2Adamax/dense_10/bias/v
2:0А2%Adamax/batch_normalization_23/gamma/v
1:/А2$Adamax/batch_normalization_23/beta/v
):'	А
2Adamax/dense_11/kernel/v
": 
2Adamax/dense_11/bias/v╙
!__inference__wrapped_model_347823н4 ()*+23;<=>RS[\]^lmuvwxТУЮЫЭЬмн╕╡╖╢╞╟@в=
6в3
1К.
conv2d_12_input         
к "3к0
.
dense_11"К
dense_11         
э
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_349987Ц()*+MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ э
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_350005Ц()*+MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ┼
7__inference_batch_normalization_18_layer_call_fn_349956Й()*+MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ┼
7__inference_batch_normalization_18_layer_call_fn_349969Й()*+MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            э
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_350069Ц;<=>MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ э
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_350087Ц;<=>MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ┼
7__inference_batch_normalization_19_layer_call_fn_350038Й;<=>MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ┼
7__inference_batch_normalization_19_layer_call_fn_350051Й;<=>MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            э
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_350188Ц[\]^MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ э
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_350206Ц[\]^MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ┼
7__inference_batch_normalization_20_layer_call_fn_350157Й[\]^MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ┼
7__inference_batch_normalization_20_layer_call_fn_350170Й[\]^MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            э
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_350297ЦuvwxMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ э
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_350315ЦuvwxMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ┼
7__inference_batch_normalization_21_layer_call_fn_350266ЙuvwxMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @┼
7__inference_batch_normalization_21_layer_call_fn_350279ЙuvwxMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @╛
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_350429hЮЫЭЬ4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ ╛
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_350463hЭЮЫЬ4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Ц
7__inference_batch_normalization_22_layer_call_fn_350396[ЮЫЭЬ4в1
*в'
!К
inputs         А
p 
к "К         АЦ
7__inference_batch_normalization_22_layer_call_fn_350409[ЭЮЫЬ4в1
*в'
!К
inputs         А
p
к "К         А╛
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_350556h╕╡╖╢4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ ╛
R__inference_batch_normalization_23_layer_call_and_return_conditional_losses_350590h╖╕╡╢4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Ц
7__inference_batch_normalization_23_layer_call_fn_350523[╕╡╖╢4в1
*в'
!К
inputs         А
p 
к "К         АЦ
7__inference_batch_normalization_23_layer_call_fn_350536[╖╕╡╢4в1
*в'
!К
inputs         А
p
к "К         А╡
E__inference_conv2d_12_layer_call_and_return_conditional_losses_349943l 7в4
-в*
(К%
inputs         
к "-в*
#К 
0          
Ъ Н
*__inference_conv2d_12_layer_call_fn_349932_ 7в4
-в*
(К%
inputs         
к " К          ╡
E__inference_conv2d_13_layer_call_and_return_conditional_losses_350025l237в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ Н
*__inference_conv2d_13_layer_call_fn_350014_237в4
-в*
(К%
inputs          
к " К          ╡
E__inference_conv2d_14_layer_call_and_return_conditional_losses_350144lRS7в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ Н
*__inference_conv2d_14_layer_call_fn_350133_RS7в4
-в*
(К%
inputs          
к " К          ╡
E__inference_conv2d_15_layer_call_and_return_conditional_losses_350253llm7в4
-в*
(К%
inputs          
к "-в*
#К 
0         @
Ъ Н
*__inference_conv2d_15_layer_call_fn_350242_lm7в4
-в*
(К%
inputs          
к " К         @и
D__inference_dense_10_layer_call_and_return_conditional_losses_350510`мн0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ А
)__inference_dense_10_layer_call_fn_350499Sмн0в-
&в#
!К
inputs         А
к "К         Аз
D__inference_dense_11_layer_call_and_return_conditional_losses_350637_╞╟0в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ 
)__inference_dense_11_layer_call_fn_350626R╞╟0в-
&в#
!К
inputs         А
к "К         
з
C__inference_dense_9_layer_call_and_return_conditional_losses_350383`ТУ0в-
&в#
!К
inputs         └
к "&в#
К
0         А
Ъ 
(__inference_dense_9_layer_call_fn_350372SТУ0в-
&в#
!К
inputs         └
к "К         А╢
F__inference_dropout_15_layer_call_and_return_conditional_losses_350112l;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ ╢
F__inference_dropout_15_layer_call_and_return_conditional_losses_350124l;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ О
+__inference_dropout_15_layer_call_fn_350102_;в8
1в.
(К%
inputs          
p 
к " К          О
+__inference_dropout_15_layer_call_fn_350107_;в8
1в.
(К%
inputs          
p
к " К          ╢
F__inference_dropout_16_layer_call_and_return_conditional_losses_350221l;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ ╢
F__inference_dropout_16_layer_call_and_return_conditional_losses_350233l;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ О
+__inference_dropout_16_layer_call_fn_350211_;в8
1в.
(К%
inputs          
p 
к " К          О
+__inference_dropout_16_layer_call_fn_350216_;в8
1в.
(К%
inputs          
p
к " К          ╢
F__inference_dropout_17_layer_call_and_return_conditional_losses_350340l;в8
1в.
(К%
inputs         @
p 
к "-в*
#К 
0         @
Ъ ╢
F__inference_dropout_17_layer_call_and_return_conditional_losses_350352l;в8
1в.
(К%
inputs         @
p
к "-в*
#К 
0         @
Ъ О
+__inference_dropout_17_layer_call_fn_350330_;в8
1в.
(К%
inputs         @
p 
к " К         @О
+__inference_dropout_17_layer_call_fn_350335_;в8
1в.
(К%
inputs         @
p
к " К         @и
F__inference_dropout_18_layer_call_and_return_conditional_losses_350478^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ и
F__inference_dropout_18_layer_call_and_return_conditional_losses_350490^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ А
+__inference_dropout_18_layer_call_fn_350468Q4в1
*в'
!К
inputs         А
p 
к "К         АА
+__inference_dropout_18_layer_call_fn_350473Q4в1
*в'
!К
inputs         А
p
к "К         Аи
F__inference_dropout_19_layer_call_and_return_conditional_losses_350605^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ и
F__inference_dropout_19_layer_call_and_return_conditional_losses_350617^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ А
+__inference_dropout_19_layer_call_fn_350595Q4в1
*в'
!К
inputs         А
p 
к "К         АА
+__inference_dropout_19_layer_call_fn_350600Q4в1
*в'
!К
inputs         А
p
к "К         Ак
E__inference_flatten_3_layer_call_and_return_conditional_losses_350363a7в4
-в*
(К%
inputs         @
к "&в#
К
0         └
Ъ В
*__inference_flatten_3_layer_call_fn_350357T7в4
-в*
(К%
inputs         @
к "К         └ю
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_350097ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_6_layer_call_fn_350092СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_350325ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_7_layer_call_fn_350320СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ї
H__inference_sequential_3_layer_call_and_return_conditional_losses_349208з4 ()*+23;<=>RS[\]^lmuvwxТУЮЫЭЬмн╕╡╖╢╞╟HвE
>в;
1К.
conv2d_12_input         
p 

 
к "%в"
К
0         

Ъ Ї
H__inference_sequential_3_layer_call_and_return_conditional_losses_349309з4 ()*+23;<=>RS[\]^lmuvwxТУЭЮЫЬмн╖╕╡╢╞╟HвE
>в;
1К.
conv2d_12_input         
p

 
к "%в"
К
0         

Ъ ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_349627Ю4 ()*+23;<=>RS[\]^lmuvwxТУЮЫЭЬмн╕╡╖╢╞╟?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         

Ъ ы
H__inference_sequential_3_layer_call_and_return_conditional_losses_349840Ю4 ()*+23;<=>RS[\]^lmuvwxТУЭЮЫЬмн╖╕╡╢╞╟?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         

Ъ ╠
-__inference_sequential_3_layer_call_fn_348572Ъ4 ()*+23;<=>RS[\]^lmuvwxТУЮЫЭЬмн╕╡╖╢╞╟HвE
>в;
1К.
conv2d_12_input         
p 

 
к "К         
╠
-__inference_sequential_3_layer_call_fn_349107Ъ4 ()*+23;<=>RS[\]^lmuvwxТУЭЮЫЬмн╖╕╡╢╞╟HвE
>в;
1К.
conv2d_12_input         
p

 
к "К         
├
-__inference_sequential_3_layer_call_fn_349396С4 ()*+23;<=>RS[\]^lmuvwxТУЮЫЭЬмн╕╡╖╢╞╟?в<
5в2
(К%
inputs         
p 

 
к "К         
├
-__inference_sequential_3_layer_call_fn_349477С4 ()*+23;<=>RS[\]^lmuvwxТУЭЮЫЬмн╖╕╡╢╞╟?в<
5в2
(К%
inputs         
p

 
к "К         
щ
$__inference_signature_wrapper_349923└4 ()*+23;<=>RS[\]^lmuvwxТУЮЫЭЬмн╕╡╖╢╞╟SвP
в 
IкF
D
conv2d_12_input1К.
conv2d_12_input         "3к0
.
dense_11"К
dense_11         
