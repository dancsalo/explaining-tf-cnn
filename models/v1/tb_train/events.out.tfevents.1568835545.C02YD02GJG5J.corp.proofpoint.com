       �K"	  @��`�Abrain.Event:2~q�m�d     R���	�WN��`�A"��
z
input_1Placeholder*$
shape:���������||*
dtype0*/
_output_shapes
:���������||
�
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"            * 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�� �* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *�� >* 
_class
loc:@conv2d/kernel*
dtype0*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container *
shape:*
dtype0*
_output_shapes
: 
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
�
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
dtype0* 
_class
loc:@conv2d/kernel
�
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:* 
_class
loc:@conv2d/kernel
�
conv2d/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:
�
conv2d/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_nameconv2d/bias*
_class
loc:@conv2d/bias*
	container 
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 

conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
_class
loc:@conv2d/bias*
dtype0
�
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:
e
conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:
�
conv2d/Conv2DConv2Dinput_1conv2d/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������zz
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:
�
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������zz
�
*batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  �?*,
_class"
 loc:@batch_normalization/gamma
�
batch_normalization/gammaVarHandleOp*
dtype0*
_output_shapes
: **
shared_namebatch_normalization/gamma*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:
�
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
�
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*,
_class"
 loc:@batch_normalization/gamma*
dtype0
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
:
�
*batch_normalization/beta/Initializer/zerosConst*
valueB*    *+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
�
batch_normalization/betaVarHandleOp*
dtype0*
_output_shapes
: *)
shared_namebatch_normalization/beta*+
_class!
loc:@batch_normalization/beta*
	container *
shape:
�
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
�
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*+
_class!
loc:@batch_normalization/beta*
dtype0
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
�
1batch_normalization/moving_mean/Initializer/zerosConst*
valueB*    *2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
:
�
batch_normalization/moving_meanVarHandleOp*0
shared_name!batch_normalization/moving_mean*2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
: 
�
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
�
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean
�
4batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  �?*6
_class,
*(loc:@batch_normalization/moving_variance
�
#batch_normalization/moving_varianceVarHandleOp*4
shared_name%#batch_normalization/moving_variance*6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
: 
�
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
�
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
_output_shapes
: *
shape: 
x
batch_normalization/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
q
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
_output_shapes
: *
T0

o
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
_output_shapes
: *
T0

c
 batch_normalization/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
.batch_normalization/cond/ReadVariableOp/SwitchSwitchbatch_normalization/gamma batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: : 
�
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: : 
�
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:1'batch_normalization/cond/ReadVariableOp)batch_normalization/cond/ReadVariableOp_1batch_normalization/cond/Const batch_normalization/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������zz::::*
is_training(
�
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchconv2d/BiasAdd batch_normalization/cond/pred_id*
T0*!
_class
loc:@conv2d/BiasAdd*J
_output_shapes8
6:���������zz:���������zz
�
)batch_normalization/cond/ReadVariableOp_2ReadVariableOp0batch_normalization/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
0batch_normalization/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization/gamma batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: : 
�
)batch_normalization/cond/ReadVariableOp_3ReadVariableOp0batch_normalization/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
0batch_normalization/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
_output_shapes
: : *
T0*+
_class!
loc:@batch_normalization/beta
�
8batch_normalization/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOp?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitchbatch_normalization/moving_mean batch_normalization/cond/pred_id*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: : 
�
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpAbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch#batch_normalization/moving_variance batch_normalization/cond/pred_id*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: : 
�
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch)batch_normalization/cond/ReadVariableOp_2)batch_normalization/cond/ReadVariableOp_38batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������zz::::*
is_training( 
�
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchconv2d/BiasAdd batch_normalization/cond/pred_id*J
_output_shapes8
6:���������zz:���������zz*
T0*!
_class
loc:@conv2d/BiasAdd
�
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������zz: 
�
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
z
!batch_normalization/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization/cond_1/switch_tIdentity#batch_normalization/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization/cond_1/switch_fIdentity!batch_normalization/cond_1/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
 batch_normalization/cond_1/ConstConst$^batch_normalization/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *�p}?
�
"batch_normalization/cond_1/Const_1Const$^batch_normalization/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 batch_normalization/cond_1/MergeMerge"batch_normalization/cond_1/Const_1 batch_normalization/cond_1/Const*
T0*
N*
_output_shapes
: : 
�
)batch_normalization/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*2
_class(
&$loc:@batch_normalization/moving_mean
�
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x batch_normalization/cond_1/Merge*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: 
�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
�
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp batch_normalization/cond/Merge_1*
_output_shapes
:*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
�
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
: 
�
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x batch_normalization/cond_1/Merge*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: 
�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
�
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp batch_normalization/cond/Merge_2*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
�
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance
�
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *
�*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *
>*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 *
dtype0*&
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
�
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
dtype0
�
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:
�
conv2d_1/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:
�
conv2d_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
	container *
shape:
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
�
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros* 
_class
loc:@conv2d_1/bias*
dtype0
�
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_1/bias
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:
�
conv2d_1/Conv2DConv2Dbatch_normalization/cond/Mergeconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:���������<<*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:
�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������<<
�
,batch_normalization_1/gamma/Initializer/onesConst*
valueB*  �?*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
�
batch_normalization_1/gammaVarHandleOp*
shape:*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container 
�
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
�
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
�
,batch_normalization_1/beta/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:
�
batch_normalization_1/betaVarHandleOp*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape:*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_1/beta
�
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
�
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB*    *4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
�
!batch_normalization_1/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:
�
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
�
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
6batch_normalization_1/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  �?*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
%batch_normalization_1/moving_varianceVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
�
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_1/moving_variance
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_1/cond/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_1/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:1)batch_normalization_1/cond/ReadVariableOp+batch_normalization_1/cond/ReadVariableOp_1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������<<::::*
is_training(
�
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchconv2d_1/BiasAdd"batch_normalization_1/cond/pred_id*
T0*#
_class
loc:@conv2d_1/BiasAdd*J
_output_shapes8
6:���������<<:���������<<
�
+batch_normalization_1/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_1/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
�
+batch_normalization_1/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_1/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: : 
�
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch+batch_normalization_1/cond/ReadVariableOp_2+batch_normalization_1/cond/ReadVariableOp_3:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������<<::::*
is_training( 
�
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchconv2d_1/BiasAdd"batch_normalization_1/cond/pred_id*
T0*#
_class
loc:@conv2d_1/BiasAdd*J
_output_shapes8
6:���������<<:���������<<
�
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*
N*1
_output_shapes
:���������<<: *
T0
�
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
|
#batch_normalization_1/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_1/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
T0*
N*
_output_shapes
: : 
�
+batch_normalization_1/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x"batch_normalization_1/cond_1/Merge*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp"batch_normalization_1/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x"batch_normalization_1/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp"batch_normalization_1/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
�
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
�
.conv2d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *HY�*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *HY>*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_2/kernel*
seed2 
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
conv2d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container *
shape:
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
�
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
dtype0
�
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*&
_output_shapes
:
�
conv2d_2/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:
�
conv2d_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container *
shape:
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
�
conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros* 
_class
loc:@conv2d_2/bias*
dtype0
�
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_2/bias
g
conv2d_2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:
�
conv2d_2/Conv2DConv2D batch_normalization_1/cond/Mergeconv2d_2/Conv2D/ReadVariableOp*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������*
	dilations
*
T0
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:
�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
,batch_normalization_2/gamma/Initializer/onesConst*
valueB*  �?*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
�
batch_normalization_2/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
�
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
�
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0*.
_class$
" loc:@batch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
�
,batch_normalization_2/beta/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
�
batch_normalization_2/betaVarHandleOp*+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes
: 
�
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
�
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
dtype0*-
_class#
!loc:@batch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*
valueB*    *4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
�
!batch_normalization_2/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_2/moving_mean
�
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
�
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
�
6batch_normalization_2/moving_variance/Initializer/onesConst*
valueB*  �?*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
�
%batch_normalization_2/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance
�
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
�
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
)batch_normalization_2/cond/ReadVariableOpReadVariableOp2batch_normalization_2/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_2/cond/ReadVariableOp/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
+batch_normalization_2/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_2/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_2/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:1)batch_normalization_2/cond/ReadVariableOp+batch_normalization_2/cond/ReadVariableOp_1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
�
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchconv2d_2/BiasAdd"batch_normalization_2/cond/pred_id*
T0*#
_class
loc:@conv2d_2/BiasAdd*J
_output_shapes8
6:���������:���������
�
+batch_normalization_2/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_2/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
+batch_normalization_2/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_2/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_2/moving_mean"batch_normalization_2/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: : 
�
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_2/moving_variance"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: : 
�
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch+batch_normalization_2/cond/ReadVariableOp_2+batch_normalization_2/cond/ReadVariableOp_3:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchconv2d_2/BiasAdd"batch_normalization_2/cond/pred_id*
T0*#
_class
loc:@conv2d_2/BiasAdd*J
_output_shapes8
6:���������:���������
�
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
|
#batch_normalization_2/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_2/cond_1/switch_tIdentity%batch_normalization_2/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_2/cond_1/switch_fIdentity#batch_normalization_2/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_2/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_2/cond_1/ConstConst&^batch_normalization_2/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *�p}?
�
$batch_normalization_2/cond_1/Const_1Const&^batch_normalization_2/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"batch_normalization_2/cond_1/MergeMerge$batch_normalization_2/cond_1/Const_1"batch_normalization_2/cond_1/Const*
N*
_output_shapes
: : *
T0
�
+batch_normalization_2/AssignMovingAvg/sub/xConst*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
: 
�
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x"batch_normalization_2/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp"batch_normalization_2/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x"batch_normalization_2/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp"batch_normalization_2/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
�
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            *"
_class
loc:@conv2d_3/kernel
�
.conv2d_3/kernel/Initializer/random_uniform/minConst*
valueB
 *��*"
_class
loc:@conv2d_3/kernel*
dtype0*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�>*"
_class
loc:@conv2d_3/kernel
�
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_3/kernel
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
conv2d_3/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
	container *
shape:
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
�
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
dtype0
�
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:
�
conv2d_3/bias/Initializer/zerosConst*
valueB*    * 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:
�
conv2d_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container *
shape:
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
�
conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros* 
_class
loc:@conv2d_3/bias*
dtype0
�
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:
�
conv2d_3/Conv2DConv2D batch_normalization_2/cond/Mergeconv2d_3/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:���������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
i
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:
�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
,batch_normalization_3/gamma/Initializer/onesConst*
valueB*  �?*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
�
batch_normalization_3/gammaVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma
�
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
�
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
�
,batch_normalization_3/beta/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
�
batch_normalization_3/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:
�
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
�
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_3/beta*
dtype0
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes
:*-
_class#
!loc:@batch_normalization_3/beta
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*
valueB*    *4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
�
!batch_normalization_3/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:
�
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
�
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
6batch_normalization_3/moving_variance/Initializer/onesConst*
valueB*  �?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
�
%batch_normalization_3/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
: 
�
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
�
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_3/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_3/cond/ReadVariableOpReadVariableOp2batch_normalization_3/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_3/cond/ReadVariableOp/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
+batch_normalization_3/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_3/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_3/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: : 
�
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
�
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:1)batch_normalization_3/cond/ReadVariableOp+batch_normalization_3/cond/ReadVariableOp_1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(
�
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchconv2d_3/BiasAdd"batch_normalization_3/cond/pred_id*
T0*#
_class
loc:@conv2d_3/BiasAdd*J
_output_shapes8
6:���������:���������
�
+batch_normalization_3/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_3/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
+batch_normalization_3/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_3/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: : 
�
:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: : 
�
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: : 
�
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch+batch_normalization_3/cond/ReadVariableOp_2+batch_normalization_3/cond/ReadVariableOp_3:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( *
epsilon%o�:
�
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchconv2d_3/BiasAdd"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*#
_class
loc:@conv2d_3/BiasAdd
�
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
|
#batch_normalization_3/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_3/cond_1/switch_tIdentity%batch_normalization_3/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_3/cond_1/switch_fIdentity#batch_normalization_3/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_3/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_3/cond_1/ConstConst&^batch_normalization_3/cond_1/switch_t*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
$batch_normalization_3/cond_1/Const_1Const&^batch_normalization_3/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
"batch_normalization_3/cond_1/MergeMerge$batch_normalization_3/cond_1/Const_1"batch_normalization_3/cond_1/Const*
N*
_output_shapes
: : *
T0
�
+batch_normalization_3/AssignMovingAvg/sub/xConst*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
�
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x"batch_normalization_3/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp"batch_normalization_3/cond/Merge_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
�
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x"batch_normalization_3/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp"batch_normalization_3/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
�
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *��*�*"
_class
loc:@conv2d_4/kernel
�
.conv2d_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *��*>*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
�
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_4/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
conv2d_4/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
	container *
shape:
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
�
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_4/kernel*
dtype0
�
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:*"
_class
loc:@conv2d_4/kernel
�
conv2d_4/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    * 
_class
loc:@conv2d_4/bias
�
conv2d_4/biasVarHandleOp*
shared_nameconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
�
conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros* 
_class
loc:@conv2d_4/bias*
dtype0
�
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_4/bias
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:
�
conv2d_4/Conv2DConv2D batch_normalization_3/cond/Mergeconv2d_4/Conv2D/ReadVariableOp*/
_output_shapes
:���������*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
i
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:
�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
,batch_normalization_4/gamma/Initializer/onesConst*
valueB*  �?*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
_output_shapes
:
�
batch_normalization_4/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_4/gamma*.
_class$
" loc:@batch_normalization_4/gamma*
	container *
shape:
�
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
�
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_4/gamma
�
,batch_normalization_4/beta/Initializer/zerosConst*
valueB*    *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
:
�
batch_normalization_4/betaVarHandleOp*
shape:*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
	container 
�
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
�
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_4/beta*
dtype0
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_4/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *4
_class*
(&loc:@batch_normalization_4/moving_mean
�
!batch_normalization_4/moving_meanVarHandleOp*
shape:*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container 
�
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
�
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
�
6batch_normalization_4/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  �?*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
%batch_normalization_4/moving_varianceVarHandleOp*
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
	container 
�
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
�
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_4/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_4/cond/ReadVariableOpReadVariableOp2batch_normalization_4/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_4/cond/ReadVariableOp/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
�
+batch_normalization_4/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_4/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_4/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:1)batch_normalization_4/cond/ReadVariableOp+batch_normalization_4/cond/ReadVariableOp_1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training(*
epsilon%o�:
�
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchconv2d_4/BiasAdd"batch_normalization_4/cond/pred_id*
T0*#
_class
loc:@conv2d_4/BiasAdd*J
_output_shapes8
6:���������:���������
�
+batch_normalization_4/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_4/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
�
+batch_normalization_4/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_4/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_4/moving_mean"batch_normalization_4/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: : 
�
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_4/moving_variance"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch+batch_normalization_4/cond/ReadVariableOp_2+batch_normalization_4/cond/ReadVariableOp_3:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������::::*
is_training( 
�
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchconv2d_4/BiasAdd"batch_normalization_4/cond/pred_id*
T0*#
_class
loc:@conv2d_4/BiasAdd*J
_output_shapes8
6:���������:���������
�
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*
N*1
_output_shapes
:���������: *
T0
�
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
|
#batch_normalization_4/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_4/cond_1/switch_tIdentity%batch_normalization_4/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_4/cond_1/switch_fIdentity#batch_normalization_4/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_4/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_4/cond_1/ConstConst&^batch_normalization_4/cond_1/switch_t*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
$batch_normalization_4/cond_1/Const_1Const&^batch_normalization_4/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
"batch_normalization_4/cond_1/MergeMerge$batch_normalization_4/cond_1/Const_1"batch_normalization_4/cond_1/Const*
T0*
N*
_output_shapes
: : 
�
+batch_normalization_4/AssignMovingAvg/sub/xConst*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
�
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x"batch_normalization_4/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: 
�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp"batch_normalization_4/cond/Merge_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
�
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
�
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x"batch_normalization_4/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp"batch_normalization_4/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
�
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
�
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
global_average_pooling2d/MeanMean batch_normalization_4/cond/Merge/global_average_pooling2d/Mean/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims( *

Tidx0
�
global_average_pooling2d_targetPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
v
total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class

loc:@total
�
totalVarHandleOp*
shape: *
dtype0*
_output_shapes
: *
shared_nametotal*
_class

loc:@total*
	container 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: *
_class

loc:@total
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
�
countVarHandleOp*
shared_namecount*
_class

loc:@count*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
_class

loc:@count*
dtype0
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxglobal_average_pooling2d_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMax_1ArgMaxglobal_average_pooling2d/Meanmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:���������
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
y
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
�
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
[
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
�
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
m
(loss/global_average_pooling2d_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
8loss/global_average_pooling2d_loss/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
&loss/global_average_pooling2d_loss/SumSumglobal_average_pooling2d/Mean8loss/global_average_pooling2d_loss/Sum/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
�
*loss/global_average_pooling2d_loss/truedivRealDivglobal_average_pooling2d/Mean&loss/global_average_pooling2d_loss/Sum*'
_output_shapes
:���������*
T0
o
*loss/global_average_pooling2d_loss/Const_1Const*
valueB
 *���3*
dtype0*
_output_shapes
: 
m
(loss/global_average_pooling2d_loss/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&loss/global_average_pooling2d_loss/subSub(loss/global_average_pooling2d_loss/sub/x*loss/global_average_pooling2d_loss/Const_1*
T0*
_output_shapes
: 
�
8loss/global_average_pooling2d_loss/clip_by_value/MinimumMinimum*loss/global_average_pooling2d_loss/truediv&loss/global_average_pooling2d_loss/sub*
T0*'
_output_shapes
:���������
�
0loss/global_average_pooling2d_loss/clip_by_valueMaximum8loss/global_average_pooling2d_loss/clip_by_value/Minimum*loss/global_average_pooling2d_loss/Const_1*
T0*'
_output_shapes
:���������
�
&loss/global_average_pooling2d_loss/LogLog0loss/global_average_pooling2d_loss/clip_by_value*
T0*'
_output_shapes
:���������
�
&loss/global_average_pooling2d_loss/mulMulglobal_average_pooling2d_target&loss/global_average_pooling2d_loss/Log*'
_output_shapes
:���������*
T0
|
:loss/global_average_pooling2d_loss/Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
(loss/global_average_pooling2d_loss/Sum_1Sum&loss/global_average_pooling2d_loss/mul:loss/global_average_pooling2d_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
�
&loss/global_average_pooling2d_loss/NegNeg(loss/global_average_pooling2d_loss/Sum_1*#
_output_shapes
:���������*
T0
{
6loss/global_average_pooling2d_loss/weighted_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
eloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
dloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
dloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape&loss/global_average_pooling2d_loss/Neg*
T0*
out_type0*
_output_shapes
:
�
closs/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
{
sloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Rloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape&loss/global_average_pooling2d_loss/Negt^loss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Rloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/ConstConstt^loss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_likeFillRloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/ShapeRloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:���������
�
Bloss/global_average_pooling2d_loss/weighted_loss/broadcast_weightsMul6loss/global_average_pooling2d_loss/weighted_loss/ConstLloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:���������
�
4loss/global_average_pooling2d_loss/weighted_loss/MulMul&loss/global_average_pooling2d_loss/NegBloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
t
*loss/global_average_pooling2d_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
�
(loss/global_average_pooling2d_loss/Sum_2Sum4loss/global_average_pooling2d_loss/weighted_loss/Mul*loss/global_average_pooling2d_loss/Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
/loss/global_average_pooling2d_loss/num_elementsSize4loss/global_average_pooling2d_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 
�
4loss/global_average_pooling2d_loss/num_elements/CastCast/loss/global_average_pooling2d_loss/num_elements*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
m
*loss/global_average_pooling2d_loss/Const_3Const*
valueB *
dtype0*
_output_shapes
: 
�
(loss/global_average_pooling2d_loss/Sum_3Sum(loss/global_average_pooling2d_loss/Sum_2*loss/global_average_pooling2d_loss/Const_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
(loss/global_average_pooling2d_loss/valueDivNoNan(loss/global_average_pooling2d_loss/Sum_34loss/global_average_pooling2d_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
loss/mulMul
loss/mul/x(loss/global_average_pooling2d_loss/value*
T0*
_output_shapes
: "&�B���     `U,	� P��`�AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
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
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
E
AssignSubVariableOp
resource
value"dtype"
dtypetype�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
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
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%��8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
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
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�*1.14.02v1.14.0-rc1-22-gaf24dc91b5��
z
input_1Placeholder*
dtype0*/
_output_shapes
:���������||*$
shape:���������||
�
.conv2d/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2d/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
,conv2d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv2d/kernel*
valueB
 *�� �*
dtype0*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv2d/kernel*
valueB
 *�� >*
dtype0*
_output_shapes
: 
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0* 
_class
loc:@conv2d/kernel
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:
�
conv2d/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
	container *
shape:
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
�
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
dtype0* 
_class
loc:@conv2d/kernel
�
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:* 
_class
loc:@conv2d/kernel
�
conv2d/bias/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB*    *
dtype0*
_output_shapes
:
�
conv2d/biasVarHandleOp*
shared_nameconv2d/bias*
_class
loc:@conv2d/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 

conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
_class
loc:@conv2d/bias*
dtype0
�
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:
�
conv2d/Conv2DConv2Dinput_1conv2d/Conv2D/ReadVariableOp*
paddingVALID*/
_output_shapes
:���������zz*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:
�
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������zz
�
*batch_normalization/gamma/Initializer/onesConst*,
_class"
 loc:@batch_normalization/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization/gammaVarHandleOp*
dtype0*
_output_shapes
: **
shared_namebatch_normalization/gamma*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:
�
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
�
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*,
_class"
 loc:@batch_normalization/gamma*
dtype0
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*,
_class"
 loc:@batch_normalization/gamma*
dtype0*
_output_shapes
:
�
*batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*+
_class!
loc:@batch_normalization/beta*
valueB*    
�
batch_normalization/betaVarHandleOp*)
shared_namebatch_normalization/beta*+
_class!
loc:@batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes
: 
�
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
�
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*+
_class!
loc:@batch_normalization/beta*
dtype0
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*+
_class!
loc:@batch_normalization/beta*
dtype0*
_output_shapes
:
�
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization/moving_meanVarHandleOp*0
shared_name!batch_normalization/moving_mean*2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
: 
�
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
�
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
:
�
4batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB*  �?
�
#batch_normalization/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *4
shared_name%#batch_normalization/moving_variance*6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:
�
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
�
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
:
\
keras_learning_phase/inputConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
shape: *
dtype0
*
_output_shapes
: 
x
batch_normalization/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
q
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
_output_shapes
: *
T0

o
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
_output_shapes
: *
T0

c
 batch_normalization/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
.batch_normalization/cond/ReadVariableOp/SwitchSwitchbatch_normalization/gamma batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: : 
�
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: : 
�
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:1'batch_normalization/cond/ReadVariableOp)batch_normalization/cond/ReadVariableOp_1batch_normalization/cond/Const batch_normalization/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*
is_training(*G
_output_shapes5
3:���������zz::::
�
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchconv2d/BiasAdd batch_normalization/cond/pred_id*
T0*!
_class
loc:@conv2d/BiasAdd*J
_output_shapes8
6:���������zz:���������zz
�
)batch_normalization/cond/ReadVariableOp_2ReadVariableOp0batch_normalization/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
0batch_normalization/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization/gamma batch_normalization/cond/pred_id*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: : 
�
)batch_normalization/cond/ReadVariableOp_3ReadVariableOp0batch_normalization/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
0batch_normalization/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: : 
�
8batch_normalization/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOp?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitchbatch_normalization/moving_mean batch_normalization/cond/pred_id*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: : 
�
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpAbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch#batch_normalization/moving_variance batch_normalization/cond/pred_id*
_output_shapes
: : *
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch)batch_normalization/cond/ReadVariableOp_2)batch_normalization/cond/ReadVariableOp_38batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%o�:*
T0*
data_formatNHWC*
is_training( *G
_output_shapes5
3:���������zz::::
�
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchconv2d/BiasAdd batch_normalization/cond/pred_id*
T0*!
_class
loc:@conv2d/BiasAdd*J
_output_shapes8
6:���������zz:���������zz
�
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������zz: 
�
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
z
!batch_normalization/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization/cond_1/switch_tIdentity#batch_normalization/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization/cond_1/switch_fIdentity!batch_normalization/cond_1/Switch*
_output_shapes
: *
T0

e
"batch_normalization/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

�
 batch_normalization/cond_1/ConstConst$^batch_normalization/cond_1/switch_t*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
"batch_normalization/cond_1/Const_1Const$^batch_normalization/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 batch_normalization/cond_1/MergeMerge"batch_normalization/cond_1/Const_1 batch_normalization/cond_1/Const*
T0*
N*
_output_shapes
: : 
�
)batch_normalization/AssignMovingAvg/sub/xConst*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x batch_normalization/cond_1/Merge*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: 
�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
�
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp batch_normalization/cond/Merge_1*
_output_shapes
:*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
�
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
�
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x batch_normalization/cond_1/Merge*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
�
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp batch_normalization/cond/Merge_2*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
�
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *
�*
dtype0*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *
>*
dtype0*
_output_shapes
: 
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_1/kernel*
seed2 
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_1/kernel
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:
�
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
�
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
dtype0
�
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:
�
conv2d_1/bias/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
conv2d_1/biasVarHandleOp* 
_class
loc:@conv2d_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
�
conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros* 
_class
loc:@conv2d_1/bias*
dtype0
�
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:
�
conv2d_1/Conv2DConv2Dbatch_normalization/cond/Mergeconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:���������<<*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:
�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:���������<<*
T0
�
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_1/gammaVarHandleOp*,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape:*
dtype0*
_output_shapes
: 
�
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
�
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:
�
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_1/betaVarHandleOp*
shape:*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
	container 
�
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
�
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_1/moving_meanVarHandleOp*2
shared_name#!batch_normalization_1/moving_mean*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
: 
�
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
�
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_1/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape:
�
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
�
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_1/cond/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
�
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_1/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
�
 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:1)batch_normalization_1/cond/ReadVariableOp+batch_normalization_1/cond/ReadVariableOp_1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*
is_training(*G
_output_shapes5
3:���������<<::::
�
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchconv2d_1/BiasAdd"batch_normalization_1/cond/pred_id*
T0*#
_class
loc:@conv2d_1/BiasAdd*J
_output_shapes8
6:���������<<:���������<<
�
+batch_normalization_1/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_1/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
�
+batch_normalization_1/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_1/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch+batch_normalization_1/cond/ReadVariableOp_2+batch_normalization_1/cond/ReadVariableOp_3:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*
is_training( *G
_output_shapes5
3:���������<<::::*
epsilon%o�:
�
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchconv2d_1/BiasAdd"batch_normalization_1/cond/pred_id*
T0*#
_class
loc:@conv2d_1/BiasAdd*J
_output_shapes8
6:���������<<:���������<<
�
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*
N*1
_output_shapes
:���������<<: *
T0
�
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:: 
|
#batch_normalization_1/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

y
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_1/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *�p}?
�
$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
N*
_output_shapes
: : *
T0
�
+batch_normalization_1/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x"batch_normalization_1/cond_1/Merge*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp"batch_normalization_1/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
�
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
�
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB
 *  �?
�
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x"batch_normalization_1/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp"batch_normalization_1/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
�
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
�
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_2/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *HY�
�
.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *HY>*
dtype0*
_output_shapes
: 
�
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_2/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
�
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:
�
conv2d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container *
shape:
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
�
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_2/kernel
�
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:*"
_class
loc:@conv2d_2/kernel
�
conv2d_2/bias/Initializer/zerosConst* 
_class
loc:@conv2d_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
conv2d_2/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
	container *
shape:
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 
�
conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_2/bias
�
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
:
g
conv2d_2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:
�
conv2d_2/Conv2DConv2D batch_normalization_1/cond/Mergeconv2d_2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������
i
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:
�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
data_formatNHWC*/
_output_shapes
:���������*
T0
�
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_2/gammaVarHandleOp*
shape:*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
	container 
�
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
�
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
_output_shapes
:
�
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_2/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:
�
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
�
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_2/beta*
dtype0
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    
�
!batch_normalization_2/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:
�
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
�
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_2/moving_varianceVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
�
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_2/cond/ReadVariableOpReadVariableOp2batch_normalization_2/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_2/cond/ReadVariableOp/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
+batch_normalization_2/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_2/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_2/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:1)batch_normalization_2/cond/ReadVariableOp+batch_normalization_2/cond/ReadVariableOp_1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
T0*
data_formatNHWC*
is_training(*G
_output_shapes5
3:���������::::*
epsilon%o�:
�
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchconv2d_2/BiasAdd"batch_normalization_2/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*#
_class
loc:@conv2d_2/BiasAdd
�
+batch_normalization_2/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_2/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: : 
�
+batch_normalization_2/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_2/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : 
�
:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_2/moving_mean"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_2/moving_variance"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch+batch_normalization_2/cond/ReadVariableOp_2+batch_normalization_2/cond/ReadVariableOp_3:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*
is_training( *G
_output_shapes5
3:���������::::*
epsilon%o�:
�
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchconv2d_2/BiasAdd"batch_normalization_2/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*#
_class
loc:@conv2d_2/BiasAdd
�
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
N*
_output_shapes

:: *
T0
�
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
N*
_output_shapes

:: *
T0
|
#batch_normalization_2/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_2/cond_1/switch_tIdentity%batch_normalization_2/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_2/cond_1/switch_fIdentity#batch_normalization_2/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_2/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_2/cond_1/ConstConst&^batch_normalization_2/cond_1/switch_t*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
$batch_normalization_2/cond_1/Const_1Const&^batch_normalization_2/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"batch_normalization_2/cond_1/MergeMerge$batch_normalization_2/cond_1/Const_1"batch_normalization_2/cond_1/Const*
N*
_output_shapes
: : *
T0
�
+batch_normalization_2/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x"batch_normalization_2/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp"batch_normalization_2/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
�
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x"batch_normalization_2/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: 
�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp"batch_normalization_2/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
�
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_3/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
valueB
 *��
�
.conv2d_3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *�>*
dtype0*
_output_shapes
: 
�
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
�
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
�
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:
�
conv2d_3/kernelVarHandleOp*
dtype0*
_output_shapes
: * 
shared_nameconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
	container *
shape:
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
�
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_3/kernel
�
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*&
_output_shapes
:
�
conv2d_3/bias/Initializer/zerosConst* 
_class
loc:@conv2d_3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
conv2d_3/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container *
shape:
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
�
conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros* 
_class
loc:@conv2d_3/bias*
dtype0
�
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes
:
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*&
_output_shapes
:
�
conv2d_3/Conv2DConv2D batch_normalization_2/cond/Mergeconv2d_3/Conv2D/ReadVariableOp*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:���������*
	dilations

i
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes
:
�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_3/gammaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:
�
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
�
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
_output_shapes
:
�
,batch_normalization_3/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*-
_class#
!loc:@batch_normalization_3/beta*
valueB*    
�
batch_normalization_3/betaVarHandleOp*
dtype0*
_output_shapes
: *+
shared_namebatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:
�
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
�
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_3/beta*
dtype0
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB*    
�
!batch_normalization_3/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:
�
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
�
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
�
6batch_normalization_3/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_3/moving_varianceVarHandleOp*
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container 
�
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
�
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_3/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_3/cond/ReadVariableOpReadVariableOp2batch_normalization_3/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_3/cond/ReadVariableOp/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
+batch_normalization_3/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_3/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_3/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_3/beta
�
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:1)batch_normalization_3/cond/ReadVariableOp+batch_normalization_3/cond/ReadVariableOp_1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*
epsilon%o�:*
T0*
data_formatNHWC*
is_training(*G
_output_shapes5
3:���������::::
�
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchconv2d_3/BiasAdd"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*#
_class
loc:@conv2d_3/BiasAdd
�
+batch_normalization_3/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_3/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: : 
�
+batch_normalization_3/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_3/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: : 
�
:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch+batch_normalization_3/cond/ReadVariableOp_2+batch_normalization_3/cond/ReadVariableOp_3:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*
is_training( *G
_output_shapes5
3:���������::::*
epsilon%o�:
�
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchconv2d_3/BiasAdd"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:���������:���������*
T0*#
_class
loc:@conv2d_3/BiasAdd
�
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
N*
_output_shapes

:: *
T0
|
#batch_normalization_3/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_3/cond_1/switch_tIdentity%batch_normalization_3/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_3/cond_1/switch_fIdentity#batch_normalization_3/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_3/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_3/cond_1/ConstConst&^batch_normalization_3/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *�p}?
�
$batch_normalization_3/cond_1/Const_1Const&^batch_normalization_3/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
"batch_normalization_3/cond_1/MergeMerge$batch_normalization_3/cond_1/Const_1"batch_normalization_3/cond_1/Const*
T0*
N*
_output_shapes
: : 
�
+batch_normalization_3/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x"batch_normalization_3/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp"batch_normalization_3/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:
�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x"batch_normalization_3/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp"batch_normalization_3/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
�
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
�
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
.conv2d_4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *��*�*
dtype0*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
valueB
 *��*>
�
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_4/kernel*
seed2 *
dtype0*&
_output_shapes
:*

seed 
�
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: 
�
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
�
conv2d_4/kernelVarHandleOp*
shape:*
dtype0*
_output_shapes
: * 
shared_nameconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
	container 
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
�
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_4/kernel*
dtype0
�
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*&
_output_shapes
:
�
conv2d_4/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_4/bias*
valueB*    
�
conv2d_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shared_nameconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
	container *
shape:
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
�
conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros* 
_class
loc:@conv2d_4/bias*
dtype0
�
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
dtype0*
_output_shapes
:
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*&
_output_shapes
:
�
conv2d_4/Conv2DConv2D batch_normalization_3/cond/Mergeconv2d_4/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*/
_output_shapes
:���������
i
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes
:
�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������
�
,batch_normalization_4/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_4/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_4/gammaVarHandleOp*
shape:*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_4/gamma*.
_class$
" loc:@batch_normalization_4/gamma*
	container 
�
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
�
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
dtype0*.
_class$
" loc:@batch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_4/gamma
�
,batch_normalization_4/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_4/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_4/betaVarHandleOp*+
shared_namebatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
	container *
shape:*
dtype0*
_output_shapes
: 
�
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
�
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_4/beta*
dtype0
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
:
�
3batch_normalization_4/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_4/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
shape:
�
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
�
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
dtype0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
�
6batch_normalization_4/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_4/moving_varianceVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
�
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_4/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
)batch_normalization_4/cond/ReadVariableOpReadVariableOp2batch_normalization_4/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:
�
0batch_normalization_4/cond/ReadVariableOp/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
�
+batch_normalization_4/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_4/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:
�
2batch_normalization_4/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
�
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:1)batch_normalization_4/cond/ReadVariableOp+batch_normalization_4/cond/ReadVariableOp_1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
data_formatNHWC*
is_training(*G
_output_shapes5
3:���������::::*
epsilon%o�:*
T0
�
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchconv2d_4/BiasAdd"batch_normalization_4/cond/pred_id*
T0*#
_class
loc:@conv2d_4/BiasAdd*J
_output_shapes8
6:���������:���������
�
+batch_normalization_4/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_4/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: : 
�
+batch_normalization_4/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:
�
2batch_normalization_4/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: : 
�
:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:
�
Abatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_4/moving_mean"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
�
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_4/moving_variance"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch+batch_normalization_4/cond/ReadVariableOp_2+batch_normalization_4/cond/ReadVariableOp_3:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*
is_training( *G
_output_shapes5
3:���������::::*
epsilon%o�:
�
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchconv2d_4/BiasAdd"batch_normalization_4/cond/pred_id*
T0*#
_class
loc:@conv2d_4/BiasAdd*J
_output_shapes8
6:���������:���������
�
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*
T0*
N*1
_output_shapes
:���������: 
�
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
T0*
N*
_output_shapes

:: 
�
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
N*
_output_shapes

:: *
T0
|
#batch_normalization_4/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_4/cond_1/switch_tIdentity%batch_normalization_4/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_4/cond_1/switch_fIdentity#batch_normalization_4/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_4/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
�
"batch_normalization_4/cond_1/ConstConst&^batch_normalization_4/cond_1/switch_t*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
�
$batch_normalization_4/cond_1/Const_1Const&^batch_normalization_4/cond_1/switch_f*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
"batch_normalization_4/cond_1/MergeMerge$batch_normalization_4/cond_1/Const_1"batch_normalization_4/cond_1/Const*
N*
_output_shapes
: : *
T0
�
+batch_normalization_4/AssignMovingAvg/sub/xConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB
 *  �?
�
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x"batch_normalization_4/cond_1/Merge*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
�
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp"batch_normalization_4/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
�
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
�
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
�
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x"batch_normalization_4/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
�
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp"batch_normalization_4/cond/Merge_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
�
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
�
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
�
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
�
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
�
global_average_pooling2d/MeanMean batch_normalization_4/cond/Merge/global_average_pooling2d/Mean/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims( 
�
global_average_pooling2d_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
v
total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class

loc:@total*
valueB
 *    
�
totalVarHandleOp*
shared_nametotal*
_class

loc:@total*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
g
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
_class

loc:@total*
dtype0
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class

loc:@count*
valueB
 *    
�
countVarHandleOp*
dtype0*
_output_shapes
: *
shared_namecount*
_class

loc:@count*
	container *
shape: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
g
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0*
_class

loc:@count
q
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: *
_class

loc:@count
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxglobal_average_pooling2d_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMax_1ArgMaxglobal_average_pooling2d/Meanmetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:���������
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
y
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
�
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
[
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
�
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
�
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
�
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
_output_shapes
: *
T0
m
(loss/global_average_pooling2d_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
z
8loss/global_average_pooling2d_loss/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
&loss/global_average_pooling2d_loss/SumSumglobal_average_pooling2d/Mean8loss/global_average_pooling2d_loss/Sum/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
*loss/global_average_pooling2d_loss/truedivRealDivglobal_average_pooling2d/Mean&loss/global_average_pooling2d_loss/Sum*
T0*'
_output_shapes
:���������
o
*loss/global_average_pooling2d_loss/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *���3
m
(loss/global_average_pooling2d_loss/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
&loss/global_average_pooling2d_loss/subSub(loss/global_average_pooling2d_loss/sub/x*loss/global_average_pooling2d_loss/Const_1*
T0*
_output_shapes
: 
�
8loss/global_average_pooling2d_loss/clip_by_value/MinimumMinimum*loss/global_average_pooling2d_loss/truediv&loss/global_average_pooling2d_loss/sub*'
_output_shapes
:���������*
T0
�
0loss/global_average_pooling2d_loss/clip_by_valueMaximum8loss/global_average_pooling2d_loss/clip_by_value/Minimum*loss/global_average_pooling2d_loss/Const_1*'
_output_shapes
:���������*
T0
�
&loss/global_average_pooling2d_loss/LogLog0loss/global_average_pooling2d_loss/clip_by_value*
T0*'
_output_shapes
:���������
�
&loss/global_average_pooling2d_loss/mulMulglobal_average_pooling2d_target&loss/global_average_pooling2d_loss/Log*
T0*'
_output_shapes
:���������
|
:loss/global_average_pooling2d_loss/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
(loss/global_average_pooling2d_loss/Sum_1Sum&loss/global_average_pooling2d_loss/mul:loss/global_average_pooling2d_loss/Sum_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
�
&loss/global_average_pooling2d_loss/NegNeg(loss/global_average_pooling2d_loss/Sum_1*
T0*#
_output_shapes
:���������
{
6loss/global_average_pooling2d_loss/weighted_loss/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
eloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
dloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
dloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape&loss/global_average_pooling2d_loss/Neg*
T0*
out_type0*
_output_shapes
:
�
closs/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
{
sloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
�
Rloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape&loss/global_average_pooling2d_loss/Negt^loss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
�
Rloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/ConstConstt^loss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_likeFillRloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/ShapeRloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:���������*
T0*

index_type0
�
Bloss/global_average_pooling2d_loss/weighted_loss/broadcast_weightsMul6loss/global_average_pooling2d_loss/weighted_loss/ConstLloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights/ones_like*#
_output_shapes
:���������*
T0
�
4loss/global_average_pooling2d_loss/weighted_loss/MulMul&loss/global_average_pooling2d_loss/NegBloss/global_average_pooling2d_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:���������
t
*loss/global_average_pooling2d_loss/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
�
(loss/global_average_pooling2d_loss/Sum_2Sum4loss/global_average_pooling2d_loss/weighted_loss/Mul*loss/global_average_pooling2d_loss/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
/loss/global_average_pooling2d_loss/num_elementsSize4loss/global_average_pooling2d_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 
�
4loss/global_average_pooling2d_loss/num_elements/CastCast/loss/global_average_pooling2d_loss/num_elements*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
m
*loss/global_average_pooling2d_loss/Const_3Const*
valueB *
dtype0*
_output_shapes
: 
�
(loss/global_average_pooling2d_loss/Sum_3Sum(loss/global_average_pooling2d_loss/Sum_2*loss/global_average_pooling2d_loss/Const_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
(loss/global_average_pooling2d_loss/valueDivNoNan(loss/global_average_pooling2d_loss/Sum_34loss/global_average_pooling2d_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
loss/mulMul
loss/mul/x(loss/global_average_pooling2d_loss/value*
T0*
_output_shapes
: "&"�
trainable_variables��
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
�
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
�
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
�
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
�
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
�
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
�
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
�
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
�
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08"�
cond_context܆؆
�
"batch_normalization/cond/cond_text"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_t:0 *�
batch_normalization/beta:0
 batch_normalization/cond/Const:0
"batch_normalization/cond/Const_1:0
0batch_normalization/cond/FusedBatchNorm/Switch:1
)batch_normalization/cond/FusedBatchNorm:0
)batch_normalization/cond/FusedBatchNorm:1
)batch_normalization/cond/FusedBatchNorm:2
)batch_normalization/cond/FusedBatchNorm:3
)batch_normalization/cond/FusedBatchNorm:4
0batch_normalization/cond/ReadVariableOp/Switch:1
)batch_normalization/cond/ReadVariableOp:0
2batch_normalization/cond/ReadVariableOp_1/Switch:1
+batch_normalization/cond/ReadVariableOp_1:0
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_t:0
batch_normalization/gamma:0
conv2d/BiasAdd:0H
"batch_normalization/cond/pred_id:0"batch_normalization/cond/pred_id:0O
batch_normalization/gamma:00batch_normalization/cond/ReadVariableOp/Switch:1P
batch_normalization/beta:02batch_normalization/cond/ReadVariableOp_1/Switch:1D
conv2d/BiasAdd:00batch_normalization/cond/FusedBatchNorm/Switch:1
�
$batch_normalization/cond/cond_text_1"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_f:0*�
batch_normalization/beta:0
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:0
Cbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
<batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1:0
2batch_normalization/cond/FusedBatchNorm_1/Switch:0
+batch_normalization/cond/FusedBatchNorm_1:0
+batch_normalization/cond/FusedBatchNorm_1:1
+batch_normalization/cond/FusedBatchNorm_1:2
+batch_normalization/cond/FusedBatchNorm_1:3
+batch_normalization/cond/FusedBatchNorm_1:4
2batch_normalization/cond/ReadVariableOp_2/Switch:0
+batch_normalization/cond/ReadVariableOp_2:0
2batch_normalization/cond/ReadVariableOp_3/Switch:0
+batch_normalization/cond/ReadVariableOp_3:0
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_f:0
batch_normalization/gamma:0
!batch_normalization/moving_mean:0
%batch_normalization/moving_variance:0
conv2d/BiasAdd:0H
"batch_normalization/cond/pred_id:0"batch_normalization/cond/pred_id:0f
!batch_normalization/moving_mean:0Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0l
%batch_normalization/moving_variance:0Cbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0F
conv2d/BiasAdd:02batch_normalization/cond/FusedBatchNorm_1/Switch:0P
batch_normalization/beta:02batch_normalization/cond/ReadVariableOp_3/Switch:0Q
batch_normalization/gamma:02batch_normalization/cond/ReadVariableOp_2/Switch:0
�
$batch_normalization/cond_1/cond_text$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_t:0 *�
"batch_normalization/cond_1/Const:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_t:0L
$batch_normalization/cond_1/pred_id:0$batch_normalization/cond_1/pred_id:0
�
&batch_normalization/cond_1/cond_text_1$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_f:0*�
$batch_normalization/cond_1/Const_1:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_f:0L
$batch_normalization/cond_1/pred_id:0$batch_normalization/cond_1/pred_id:0
�
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *�
batch_normalization_1/beta:0
"batch_normalization_1/cond/Const:0
$batch_normalization_1/cond/Const_1:0
2batch_normalization_1/cond/FusedBatchNorm/Switch:1
+batch_normalization_1/cond/FusedBatchNorm:0
+batch_normalization_1/cond/FusedBatchNorm:1
+batch_normalization_1/cond/FusedBatchNorm:2
+batch_normalization_1/cond/FusedBatchNorm:3
+batch_normalization_1/cond/FusedBatchNorm:4
2batch_normalization_1/cond/ReadVariableOp/Switch:1
+batch_normalization_1/cond/ReadVariableOp:0
4batch_normalization_1/cond/ReadVariableOp_1/Switch:1
-batch_normalization_1/cond/ReadVariableOp_1:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0
batch_normalization_1/gamma:0
conv2d_1/BiasAdd:0S
batch_normalization_1/gamma:02batch_normalization_1/cond/ReadVariableOp/Switch:1T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_1/Switch:1H
conv2d_1/BiasAdd:02batch_normalization_1/cond/FusedBatchNorm/Switch:1L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0
�
&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*�
batch_normalization_1/beta:0
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_1/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_1/cond/FusedBatchNorm_1:0
-batch_normalization_1/cond/FusedBatchNorm_1:1
-batch_normalization_1/cond/FusedBatchNorm_1:2
-batch_normalization_1/cond/FusedBatchNorm_1:3
-batch_normalization_1/cond/FusedBatchNorm_1:4
4batch_normalization_1/cond/ReadVariableOp_2/Switch:0
-batch_normalization_1/cond/ReadVariableOp_2:0
4batch_normalization_1/cond/ReadVariableOp_3/Switch:0
-batch_normalization_1/cond/ReadVariableOp_3:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
batch_normalization_1/gamma:0
#batch_normalization_1/moving_mean:0
'batch_normalization_1/moving_variance:0
conv2d_1/BiasAdd:0T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_3/Switch:0J
conv2d_1/BiasAdd:04batch_normalization_1/cond/FusedBatchNorm_1/Switch:0U
batch_normalization_1/gamma:04batch_normalization_1/cond/ReadVariableOp_2/Switch:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0j
#batch_normalization_1/moving_mean:0Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_1/moving_variance:0Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
�
&batch_normalization_1/cond_1/cond_text&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_t:0 *�
$batch_normalization_1/cond_1/Const:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_t:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
�
(batch_normalization_1/cond_1/cond_text_1&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_f:0*�
&batch_normalization_1/cond_1/Const_1:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_f:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
�
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *�
batch_normalization_2/beta:0
"batch_normalization_2/cond/Const:0
$batch_normalization_2/cond/Const_1:0
2batch_normalization_2/cond/FusedBatchNorm/Switch:1
+batch_normalization_2/cond/FusedBatchNorm:0
+batch_normalization_2/cond/FusedBatchNorm:1
+batch_normalization_2/cond/FusedBatchNorm:2
+batch_normalization_2/cond/FusedBatchNorm:3
+batch_normalization_2/cond/FusedBatchNorm:4
2batch_normalization_2/cond/ReadVariableOp/Switch:1
+batch_normalization_2/cond/ReadVariableOp:0
4batch_normalization_2/cond/ReadVariableOp_1/Switch:1
-batch_normalization_2/cond/ReadVariableOp_1:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0
batch_normalization_2/gamma:0
conv2d_2/BiasAdd:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0H
conv2d_2/BiasAdd:02batch_normalization_2/cond/FusedBatchNorm/Switch:1S
batch_normalization_2/gamma:02batch_normalization_2/cond/ReadVariableOp/Switch:1T
batch_normalization_2/beta:04batch_normalization_2/cond/ReadVariableOp_1/Switch:1
�
&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*�
batch_normalization_2/beta:0
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_2/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_2/cond/FusedBatchNorm_1:0
-batch_normalization_2/cond/FusedBatchNorm_1:1
-batch_normalization_2/cond/FusedBatchNorm_1:2
-batch_normalization_2/cond/FusedBatchNorm_1:3
-batch_normalization_2/cond/FusedBatchNorm_1:4
4batch_normalization_2/cond/ReadVariableOp_2/Switch:0
-batch_normalization_2/cond/ReadVariableOp_2:0
4batch_normalization_2/cond/ReadVariableOp_3/Switch:0
-batch_normalization_2/cond/ReadVariableOp_3:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
batch_normalization_2/gamma:0
#batch_normalization_2/moving_mean:0
'batch_normalization_2/moving_variance:0
conv2d_2/BiasAdd:0U
batch_normalization_2/gamma:04batch_normalization_2/cond/ReadVariableOp_2/Switch:0j
#batch_normalization_2/moving_mean:0Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_2/moving_variance:0Ebatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0J
conv2d_2/BiasAdd:04batch_normalization_2/cond/FusedBatchNorm_1/Switch:0T
batch_normalization_2/beta:04batch_normalization_2/cond/ReadVariableOp_3/Switch:0
�
&batch_normalization_2/cond_1/cond_text&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_t:0 *�
$batch_normalization_2/cond_1/Const:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_t:0P
&batch_normalization_2/cond_1/pred_id:0&batch_normalization_2/cond_1/pred_id:0
�
(batch_normalization_2/cond_1/cond_text_1&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_f:0*�
&batch_normalization_2/cond_1/Const_1:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_f:0P
&batch_normalization_2/cond_1/pred_id:0&batch_normalization_2/cond_1/pred_id:0
�
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *�
batch_normalization_3/beta:0
"batch_normalization_3/cond/Const:0
$batch_normalization_3/cond/Const_1:0
2batch_normalization_3/cond/FusedBatchNorm/Switch:1
+batch_normalization_3/cond/FusedBatchNorm:0
+batch_normalization_3/cond/FusedBatchNorm:1
+batch_normalization_3/cond/FusedBatchNorm:2
+batch_normalization_3/cond/FusedBatchNorm:3
+batch_normalization_3/cond/FusedBatchNorm:4
2batch_normalization_3/cond/ReadVariableOp/Switch:1
+batch_normalization_3/cond/ReadVariableOp:0
4batch_normalization_3/cond/ReadVariableOp_1/Switch:1
-batch_normalization_3/cond/ReadVariableOp_1:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0
batch_normalization_3/gamma:0
conv2d_3/BiasAdd:0T
batch_normalization_3/beta:04batch_normalization_3/cond/ReadVariableOp_1/Switch:1H
conv2d_3/BiasAdd:02batch_normalization_3/cond/FusedBatchNorm/Switch:1L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0S
batch_normalization_3/gamma:02batch_normalization_3/cond/ReadVariableOp/Switch:1
�
&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*�
batch_normalization_3/beta:0
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_3/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_3/cond/FusedBatchNorm_1:0
-batch_normalization_3/cond/FusedBatchNorm_1:1
-batch_normalization_3/cond/FusedBatchNorm_1:2
-batch_normalization_3/cond/FusedBatchNorm_1:3
-batch_normalization_3/cond/FusedBatchNorm_1:4
4batch_normalization_3/cond/ReadVariableOp_2/Switch:0
-batch_normalization_3/cond/ReadVariableOp_2:0
4batch_normalization_3/cond/ReadVariableOp_3/Switch:0
-batch_normalization_3/cond/ReadVariableOp_3:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
batch_normalization_3/gamma:0
#batch_normalization_3/moving_mean:0
'batch_normalization_3/moving_variance:0
conv2d_3/BiasAdd:0p
'batch_normalization_3/moving_variance:0Ebatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0j
#batch_normalization_3/moving_mean:0Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0J
conv2d_3/BiasAdd:04batch_normalization_3/cond/FusedBatchNorm_1/Switch:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0U
batch_normalization_3/gamma:04batch_normalization_3/cond/ReadVariableOp_2/Switch:0T
batch_normalization_3/beta:04batch_normalization_3/cond/ReadVariableOp_3/Switch:0
�
&batch_normalization_3/cond_1/cond_text&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_t:0 *�
$batch_normalization_3/cond_1/Const:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_t:0P
&batch_normalization_3/cond_1/pred_id:0&batch_normalization_3/cond_1/pred_id:0
�
(batch_normalization_3/cond_1/cond_text_1&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_f:0*�
&batch_normalization_3/cond_1/Const_1:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_f:0P
&batch_normalization_3/cond_1/pred_id:0&batch_normalization_3/cond_1/pred_id:0
�
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *�
batch_normalization_4/beta:0
"batch_normalization_4/cond/Const:0
$batch_normalization_4/cond/Const_1:0
2batch_normalization_4/cond/FusedBatchNorm/Switch:1
+batch_normalization_4/cond/FusedBatchNorm:0
+batch_normalization_4/cond/FusedBatchNorm:1
+batch_normalization_4/cond/FusedBatchNorm:2
+batch_normalization_4/cond/FusedBatchNorm:3
+batch_normalization_4/cond/FusedBatchNorm:4
2batch_normalization_4/cond/ReadVariableOp/Switch:1
+batch_normalization_4/cond/ReadVariableOp:0
4batch_normalization_4/cond/ReadVariableOp_1/Switch:1
-batch_normalization_4/cond/ReadVariableOp_1:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0
batch_normalization_4/gamma:0
conv2d_4/BiasAdd:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0H
conv2d_4/BiasAdd:02batch_normalization_4/cond/FusedBatchNorm/Switch:1S
batch_normalization_4/gamma:02batch_normalization_4/cond/ReadVariableOp/Switch:1T
batch_normalization_4/beta:04batch_normalization_4/cond/ReadVariableOp_1/Switch:1
�
&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*�
batch_normalization_4/beta:0
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_4/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_4/cond/FusedBatchNorm_1:0
-batch_normalization_4/cond/FusedBatchNorm_1:1
-batch_normalization_4/cond/FusedBatchNorm_1:2
-batch_normalization_4/cond/FusedBatchNorm_1:3
-batch_normalization_4/cond/FusedBatchNorm_1:4
4batch_normalization_4/cond/ReadVariableOp_2/Switch:0
-batch_normalization_4/cond/ReadVariableOp_2:0
4batch_normalization_4/cond/ReadVariableOp_3/Switch:0
-batch_normalization_4/cond/ReadVariableOp_3:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
batch_normalization_4/gamma:0
#batch_normalization_4/moving_mean:0
'batch_normalization_4/moving_variance:0
conv2d_4/BiasAdd:0U
batch_normalization_4/gamma:04batch_normalization_4/cond/ReadVariableOp_2/Switch:0T
batch_normalization_4/beta:04batch_normalization_4/cond/ReadVariableOp_3/Switch:0j
#batch_normalization_4/moving_mean:0Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_4/moving_variance:0Ebatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0J
conv2d_4/BiasAdd:04batch_normalization_4/cond/FusedBatchNorm_1/Switch:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0
�
&batch_normalization_4/cond_1/cond_text&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_t:0 *�
$batch_normalization_4/cond_1/Const:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_t:0P
&batch_normalization_4/cond_1/pred_id:0&batch_normalization_4/cond_1/pred_id:0
�
(batch_normalization_4/cond_1/cond_text_1&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_f:0*�
&batch_normalization_4/cond_1/Const_1:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_f:0P
&batch_normalization_4/cond_1/pred_id:0&batch_normalization_4/cond_1/pred_id:0"�'
	variables�'�'
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08
�
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
�
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
�
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign5batch_normalization/moving_mean/Read/ReadVariableOp:0(23batch_normalization/moving_mean/Initializer/zeros:0@H
�
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign9batch_normalization/moving_variance/Read/ReadVariableOp:0(26batch_normalization/moving_variance/Initializer/ones:0@H
�
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
�
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
�
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
�
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
�
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
�
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H
�
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
�
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(25batch_normalization_3/moving_mean/Initializer/zeros:0@H
�
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(28batch_normalization_3/moving_variance/Initializer/ones:0@H
�
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08
�
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
�
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
�
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(25batch_normalization_4/moving_mean/Initializer/zeros:0@H
�
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(28batch_normalization_4/moving_variance/Initializer/ones:0@H=�L�       ��2	�����`�A*


batch_loss[b�@��C       `/�#	^����`�A*

	batch_acc  �=�[�Z        )��P	����`�A*


batch_loss�M�@.��       QKD	Z���`�A*

	batch_acc  �=�L^        )��P	�&��`�A*


batch_losso��@�ϕ       QKD	c'��`�A*

	batch_accUU�=_\D�        )��P	ک?��`�A*


batch_loss8��@og�       QKD	��?��`�A*

	batch_acc  �=ʞ�<        )��P	pw��`�A*


batch_loss���@6#�D       QKD	4w��`�A*

	batch_acc  �=����        )��P	�{���`�A*


batch_loss�@M���       QKD	�|���`�A*

	batch_acc  �=#�z@        )��P	�����`�A*


batch_loss>Y�@����       QKD	g����`�A*

	batch_acc�ƛ=xDc�        )��P	�W��`�A*


batch_loss@�@c�B�       QKD	�X��`�A*

	batch_acc&L�=_Z@�        )��P	��Q��`�A*


batch_lossJ�@��<p       QKD	��Q��`�A*

	batch_acc�'�=,�i         )��P	ݛ���`�A	*


batch_lossSG�@�<��       QKD	�����`�A	*

	batch_acc�ݩ=��        )��P	�}���`�A
*


batch_loss~��@��V�       QKD	�~���`�A
*

	batch_acc���=�A�p        )��P	:���`�A*


batch_lossHv�@�gm       QKD	���`�A*

	batch_accۑ�=�/�q        )��P	�/��`�A*


batch_loss�[�@���       QKD	�/��`�A*

	batch_acc��=i��        )��P	J�e��`�A*


batch_loss��@�)�       QKD	�e��`�A*

	batch_acc#|�=LYsB        )��P	IҞ��`�A*


batch_loss�v�@�r�       QKD	Ӟ��`�A*

	batch_acc���=v�w        )��P	y����`�A*


batch_lossH��@���[       QKD	=����`�A*

	batch_acc�*�=u�p�        )��P	W���`�A*


batch_loss�K�@���R       QKD	���`�A*

	batch_acc���=�\2         )��P	�F��`�A*


batch_loss`��@�7��       QKD	TF��`�A*

	batch_acc"��=�O        )��P	�#���`�A*


batch_loss�D�@j�G       QKD	�$���`�A*

	batch_accG��=oCt        )��P	ta���`�A*


batch_lossd'AiW       QKD	5b���`�A*

	batch_acc���=.�Q        )��P	����`�A*


batch_loss��@����       QKD	�����`�A*

	batch_acc8�=F��        )��P	O�)��`�A*


batch_lossA��C�       QKD	�)��`�A*

	batch_acc�=���        )��P	�\^��`�A*


batch_loss��@5��Y       QKD	�]^��`�A*

	batch_accw
�=E��r        )��P	�r���`�A*


batch_loss���@`՝       QKD	�s���`�A*

	batch_acc��=�>)�        )��P	�:���`�A*


batch_losst�@>rc�       QKD	K;���`�A*

	batch_acc�f�=#�Rp        )��P	�N��`�A*


batch_loss�.�@�t�i       QKD	vO��`�A*

	batch_acc���=�nA�        )��P	��>��`�A*


batch_lossț�@�O�Z       QKD	ú>��`�A*

	batch_acc%��=��H�        )��P	�w��`�A*


batch_loss�A�<=�       QKD	pw��`�A*

	batch_acc��=J=�        )��P	�ج��`�A*


batch_loss��@Π:       QKD	�٬��`�A*

	batch_acc�s�=d��
        )��P	j-���`�A*


batch_loss,f�@!�dQ       QKD	0.���`�A*

	batch_acc��=(5�I        )��P	\��`�A*


batch_loss���@f�-o       QKD	+��`�A*

	batch_acc�~�=�k}�        )��P	ȯ[��`�A*


batch_loss` �@��Ul       QKD	��[��`�A*

	batch_acc�Q�=��d        )��P	�%���`�A *


batch_loss�v�@F?J�       QKD	�&���`�A *

	batch_acc%�=�S�#        )��P	G����`�A!*


batch_loss���@˅��       QKD	"����`�A!*

	batch_accH��==`R        )��P	d@	��`�A"*


batch_loss(dA��nK       QKD	�A	��`�A"*

	batch_acc�\�=�X�        )��P	I�@��`�A#*


batch_lossrIA+q+�       QKD	�@��`�A#*

	batch_acc���=o���        )��P	�x��`�A$*


batch_lossL�A8�>       QKD	ّx��`�A$*

	batch_acc���=mk��        )��P	77���`�A%*


batch_loss�]�@W��       QKD	�7���`�A%*

	batch_acc��=*�t        )��P	�����`�A&*


batch_loss���@Y7r       QKD	{����`�A&*

	batch_acc/ë={�        )��P	7�% �`�A'*


batch_lossu9A��d�       QKD	�% �`�A'*

	batch_acc��=�@�?        )��P	�T] �`�A(*


batch_loss�_A(��       QKD	eU] �`�A(*

	batch_accHɳ=���        )��P	�� �`�A)*


batch_lossv��@�T��       QKD	O� �`�A)*

	batch_acc�ɱ=O�#)        )��P	��� �`�A**


batch_lossX�@`�^-       QKD	��� �`�A**

	batch_accf#�=vz��        )��P	!:�`�A+*


batch_lossb��@��8m       QKD	�:�`�A+*

	batch_acc���=�        )��P	��@�`�A,*


batch_loss6�AZ��h       QKD	��@�`�A,*

	batch_acc���=='��        )��P	�z�`�A-*


batch_loss���@+=A�       QKD	zz�`�A-*

	batch_acc��=_�,        )��P	1���`�A.*


batch_loss ��@@���       QKD	���`�A.*

	batch_accR�=�B}�        )��P	�`��`�A/*


batch_loss���@�SW�       QKD	ka��`�A/*

	batch_acc0��=X$U�        )��P	Y-�`�A0*


batch_loss���@R5Uu       QKD	�Y-�`�A0*

	batch_acc!��=��l�        )��P	4�e�`�A1*


batch_loss�i�@�^�       QKD	�e�`�A1*

	batch_acc;�== �        )��P	I��`�A2*


batch_loss4��@�Te       QKD	��`�A2*

	batch_acc�h�=齡.        )��P	I���`�A3*


batch_loss��@[���       QKD	E���`�A3*

	batch_acc~f�= �{o        )��P	yY�`�A4*


batch_loss�%�@���       QKD	?Z�`�A4*

	batch_acc�ұ=$e_        )��P	'�T�`�A5*


batch_loss��@��Hs       QKD	�T�`�A5*

	batch_acc2F�=�l�        )��P	��`�A6*


batch_loss���@b:�       QKD	���`�A6*

	batch_acc!A�=)!��        )��P	���`�A7*


batch_loss�A��qV       QKD	���`�A7*

	batch_acc�)�=�1ת        )��P	��`�A8*


batch_loss8z�@(���       QKD	���`�A8*

	batch_acc�1�=��        )��P	�*I�`�A9*


batch_lossA �n       QKD	�+I�`�A9*

	batch_acc�i�=�?4�        )��P	����`�A:*


batch_loss���@7�u�       QKD	����`�A:*

	batch_accd��=�3��        )��P	���`�A;*


batch_loss���@�A�       QKD	���`�A;*

	batch_acc��=%=�p        )��P	-y��`�A<*


batch_lossJ�A/��	       QKD	z��`�A<*

	batch_acc�յ=J���        )��P	��9�`�A=*


batch_losslA�w�       QKD	��9�`�A=*

	batch_acc���=��        )��P	&ru�`�A>*


batch_loss˵A�<�       QKD	�ru�`�A>*

	batch_acc[1�=��6�        )��P	SB��`�A?*


batch_loss��A����       QKD	C��`�A?*

	batch_acc�Ѻ=i$��        )��P	m���`�A@*


batch_loss&��@#���       QKD	@���`�A@*

	batch_acc�`�=&�        )��P	/8.�`�AA*


batch_lossE��@�bi"       QKD	�8.�`�AA*

	batch_acc��=�O�        )��P	C�g�`�AB*


batch_loss���@����       QKD		�g�`�AB*

	batch_acc'�=$��        )��P	Ϻ��`�AC*


batch_loss�N�@���Y       QKD	����`�AC*

	batch_accމ�=g���        )��P	*V��`�AD*


batch_lossK"�@r�.       QKD	�V��`�AD*

	batch_accbr�=^{        )��P	��"�`�AE*


batch_loss�@�@Ԡ�a       QKD	5�"�`�AE*

	batch_accx�=��        )��P	Ym_�`�AF*


batch_loss���@�H2       QKD	*n_�`�AF*

	batch_acc�d�=Wt��        )��P	���`�AG*


batch_loss�'AX��       QKD	u��`�AG*

	batch_acc�f�=�:        )��P	����`�AH*


batch_lossj�A�nQ6       QKD	O���`�AH*

	batch_acc�h�=qZ�P        )��P	]��`�AI*


batch_lossѱ�@m��       QKD	.��`�AI*

	batch_acc[��=�ˁ        )��P	�*Q�`�AJ*


batch_loss�&�@B��       QKD	�+Q�`�AJ*

	batch_accC �=Iͺ�        )��P	��`�AK*


batch_loss���@�R��       QKD	���`�AK*

	batch_accX�=�:d$        )��P	���`�AL*


batch_lossشAECH       QKD	����`�AL*

	batch_accTo�=}F�        )��P	.�	�`�AM*


batch_loss���@�`�       QKD	��	�`�AM*

	batch_acc�`�=а        )��P	ܷA	�`�AN*


batch_loss6W�@�IA       QKD	��A	�`�AN*

	batch_acc݂�=        )��P	}�	�`�AO*


batch_loss���@�M:#       QKD	>�	�`�AO*

	batch_acc�=^���        )��P	Z��	�`�AP*


batch_losst �@=���       QKD	/��	�`�AP*

	batch_acc^�=��
�        )��P	���	�`�AQ*


batch_loss�f�@�ykY       QKD	���	�`�AQ*

	batch_acc'��=("A�        )��P	IJ 
�`�AR*


batch_loss�֮@���       QKD	K 
�`�AR*

	batch_accw��=����        )��P	/�N
�`�AS*


batch_loss�VA!�D       QKD	�N
�`�AS*

	batch_acc��=�b��        )��P	�!�
�`�AT*


batch_loss��A�r~w       QKD	T"�
�`�AT*

	batch_accF7�=�        )��P	���
�`�AU*


batch_loss2�@���9       QKD	���
�`�AU*

	batch_acc$@�=ϴI        )��P	W��
�`�AV*


batch_loss�rA���       QKD	5��
�`�AV*

	batch_acc�B�=#g�F        )��P	���`�AW*


batch_lossXOA����       QKD	˹�`�AW*

	batch_acc�'�=��u�        )��P	�+H�`�AX*


batch_loss׶A
���       QKD	k,H�`�AX*

	batch_acc�i�=�F0�        )��P	�-{�`�AY*


batch_loss���@v�N�       QKD	w.{�`�AY*

	batch_acc�(�=����        )��P	Z���`�AZ*


batch_loss��A��c       QKD	0���`�AZ*

	batch_accb��=���        )��P	����`�A[*


batch_loss��@5A��       QKD	����`�A[*

	batch_accK��=�h��        )��P	8��`�A\*


batch_loss8�A1�,       QKD	��`�A\*

	batch_acc�6�=��%        )��P	�_B�`�A]*


batch_lossP�A��L       QKD	�`B�`�A]*

	batch_accud�=����        )��P	zt�`�A^*


batch_lossj�A��1       QKD	�zt�`�A^*

	batch_acc�:�=��GZ        )��P	�B��`�A_*


batch_loss(7Af��o       QKD	�C��`�A_*

	batch_acc4g�=VVE        )��P	�f��`�A`*


batch_loss��A�Ce�       QKD	�g��`�A`*

	batch_acc��=�.M        )��P	�|	�`�Aa*


batch_loss�@��       QKD	�}	�`�Aa*

	batch_accTa�=�7�        )��P	<2;�`�Ab*


batch_loss��@��?       QKD	3;�`�Ab*

	batch_acc-�=/�1        )��P	�k�`�Ac*


batch_loss�AU���       QKD	(�k�`�Ac*

	batch_acc��=c��w        )��P	� ��`�Ad*


batch_loss/�A�j�       QKD	}!��`�Ad*

	batch_acc���=���        )��P	����`�Ae*


batch_loss�A��c�       QKD	����`�Ae*

	batch_acc�@�=A�s�        )��P	2 �`�Af*


batch_loss�_�@�X�~       QKD	 �`�Af*

	batch_acc�n�=�2<�        )��P	 �0�`�Ag*


batch_lossX"�@�Pɽ       QKD	��0�`�Ag*

	batch_acc�>�=YZ��        )��P	�b�`�Ah*


batch_loss�b�@I�~�       QKD	�b�`�Ah*

	batch_acc0%�=�[u        )��P	g���`�Ai*


batch_loss�߬@��       QKD	=���`�Ai*

	batch_acc���=6A@        )��P	XT��`�Aj*


batch_loss��@��       QKD	LU��`�Aj*

	batch_acc���=�
{"        )��P	:u��`�Ak*


batch_loss+�@VM<|       QKD	v��`�Ak*

	batch_acc���=x���