/***********************************************
公司：轮趣科技（东莞）有限公司
品牌：WHEELTEC
官网：wheeltec.net
淘宝店铺：shop114407458.taobao.com
速卖通: https://minibalance.aliexpress.com/store/4455017
版本：V5.0
修改时间：2022-05-05

Brand: WHEELTEC
Website: wheeltec.net
Taobao shop: shop114407458.taobao.com
Aliexpress: https://minibalance.aliexpress.com/store/4455017
Version: V5.0
Update：2022-05-05

All rights reserved
***********************************************/

#include "system.h"

//Robot software fails to flag bits
//机器人软件失能标志位
u8 Flag_Stop=0;

//The ADC value is variable in segments, depending on the number of car models. Currently there are 6 car models
//ADC值分段变量，取决于小车型号数量，目前有6种小车型号
int Divisor_Mode;

// Robot type variable
//机器人型号变量
//0=Mec_Car，1=Omni_Car，2=Akm_Car，3=Diff_Car，4=FourWheel_Car，5=Tank_Car
u8 Car_Mode=0;

//Servo control PWM value, Ackerman car special
//舵机控制PWM值，阿克曼小车专用
int Servo;

//Default speed of remote control car, unit: mm/s
//遥控小车的默认速度，单位：mm/s
float RC_Velocity=500;

//Vehicle three-axis target moving speed, unit: m/s
//小车三轴目标运动速度，单位：m/s
float Move_X, Move_Y, Move_Z;

//PID parameters of Speed control
//速度控制PID参数
float Velocity_KP=300,Velocity_KI=300;

//Smooth control of intermediate variables, dedicated to omni-directional moving cars
//平滑控制中间变量，全向移动小车专用
Smooth_Control smooth_control;

//The parameter structure of the motor
//电机的参数结构体
Motor_parameter MOTOR_A,MOTOR_B,MOTOR_C,MOTOR_D;

/************ 小车型号相关变量 **************************/
/************ Variables related to car model ************/
//Encoder accuracy
//编码器精度
float Encoder_precision;
//Wheel circumference, unit: m
//轮子周长，单位：m
float Wheel_perimeter;
//Drive wheel base, unit: m
//主动轮轮距，单位：m
float Wheel_spacing;
//The wheelbase of the front and rear axles of the trolley, unit: m
//小车前后轴的轴距，单位：m
float Axle_spacing;
//All-directional wheel turning radius, unit: m
//全向轮转弯半径，单位：m
float Omni_turn_radiaus;
/************ 小车型号相关变量 **************************/
/************ Variables related to car model ************/

//PS2 controller, Bluetooth APP, aircraft model controller, CAN communication, serial port 1, serial port 5 communication control flag bit.
//These 6 flag bits are all 0 by default, representing the serial port 3 control mode
//PS2手柄、蓝牙APP、航模手柄、CAN通信、串口1、串口5通信控制标志位。这6个标志位默认都为0，代表串口3控制模式
u8 PS2_ON_Flag=0, APP_ON_Flag=0, Remote_ON_Flag=0, CAN_ON_Flag=0, Usart1_ON_Flag, Usart5_ON_Flag;

#define CONTROL_MANUAL_SOURCE_COUNT  4U
#define CONTROL_HOST_QUEUE_DEPTH     8U
#define CONTROL_HOST_QUEUE_MASK      (CONTROL_HOST_QUEUE_DEPTH - 1U)

volatile u8 Control_State = CONTROL_STATE_STOP_LATCHED;
volatile u8 Control_Stop_Reason = CONTROL_STOP_BOOT;

static volatile u32 Manual_Request_Seq[CONTROL_MANUAL_SOURCE_COUNT];
static volatile u32 Manual_Heartbeat_Seq[CONTROL_MANUAL_SOURCE_COUNT];
static volatile u32 Manual_Disconnect_Seq[CONTROL_MANUAL_SOURCE_COUNT];
static u32 Manual_Request_Read[CONTROL_MANUAL_SOURCE_COUNT];
static u32 Manual_Heartbeat_Read[CONTROL_MANUAL_SOURCE_COUNT];
static u32 Manual_Disconnect_Read[CONTROL_MANUAL_SOURCE_COUNT];
static u16 Manual_Input_Age = 0;

static volatile u8 Host_Frame_Queue[CONTROL_HOST_QUEUE_DEPTH][RECEIVE_DATA_SIZE];
static volatile u32 Host_Frame_Tick[CONTROL_HOST_QUEUE_DEPTH];
static volatile u32 Host_Frame_Write_Seq = 0;
static volatile u32 Host_Frame_Read_Seq = 0;
static volatile u32 Host_Frame_Overflow_Seq = 0;
static volatile u32 Host_Frame_Overflow_Read = 0;
static u32 Auto_Enable_Min_Seq = 0;
static volatile u8 Ros_Heartbeat_Started = 0;
static volatile u32 Control_Tick_100Hz = 0;
static u32 Auto_Last_Command_Tick = 0;

static u8 Control_SequenceAfter(u32 sequence, u32 reference)
{
    return ((s32)(sequence - reference) > 0);
}

static u8 Control_ValidManualSource(u8 source)
{
    return (source > CONTROL_SOURCE_NONE && source < CONTROL_MANUAL_SOURCE_COUNT);
}

static u8 Control_StateManualSource(void)
{
    if (Control_State == CONTROL_STATE_MANUAL_PS2) return CONTROL_SOURCE_PS2;
    if (Control_State == CONTROL_STATE_MANUAL_APP) return CONTROL_SOURCE_APP;
    if (Control_State == CONTROL_STATE_MANUAL_REMOTE) return CONTROL_SOURCE_REMOTE;
    return CONTROL_SOURCE_NONE;
}

static void Control_SetSourceFlags(void)
{
    PS2_ON_Flag = (Control_State == CONTROL_STATE_MANUAL_PS2);
    APP_ON_Flag = (Control_State == CONTROL_STATE_MANUAL_APP);
    Remote_ON_Flag = (Control_State == CONTROL_STATE_MANUAL_REMOTE);
    CAN_ON_Flag = 0;
    Usart1_ON_Flag = 0;
    Usart5_ON_Flag = 0;
}

static void Control_DiscardManualEvents(void)
{
    u8 source;
    for (source = CONTROL_SOURCE_PS2; source < CONTROL_MANUAL_SOURCE_COUNT; source++)
    {
        Manual_Request_Read[source] = Manual_Request_Seq[source];
        Manual_Heartbeat_Read[source] = Manual_Heartbeat_Seq[source];
        Manual_Disconnect_Read[source] = Manual_Disconnect_Seq[source];
    }
}

static void Control_ZeroMotion(void)
{
    Move_X = 0.0f;
    Move_Y = 0.0f;
    Move_Z = 0.0f;
    Recharge_UP_Move_X = 0.0f;
    Recharge_UP_Move_Y = 0.0f;
    Recharge_UP_Move_Z = 0.0f;
    Recharge_Red_Move_X = 0.0f;
    Recharge_Red_Move_Y = 0.0f;
    Recharge_Red_Move_Z = 0.0f;
    smooth_control.VX = 0.0f;
    smooth_control.VY = 0.0f;
    smooth_control.VZ = 0.0f;
    MOTOR_A.Target = 0.0f;
    MOTOR_B.Target = 0.0f;
    MOTOR_C.Target = 0.0f;
    MOTOR_D.Target = 0.0f;
    Allow_Recharge = 0;
    nav_walk = 0;
}

static void Control_StopAndLatch(u8 reason)
{
    Control_State = CONTROL_STATE_STOP_LATCHED;
    Control_Stop_Reason = reason;
    Ros_Heartbeat_Started = 0;
    command_lost_count = COMMAND_LOST_TIMEOUT_TICKS;
    Auto_Last_Command_Tick = Control_Tick_100Hz;
    Auto_Enable_Min_Seq = Host_Frame_Write_Seq;
    Control_ZeroMotion();
    Control_SetSourceFlags();
    Control_DiscardManualEvents();
    if (GamePadInterface != 0)
        GamePadInterface->StartFlag = 0;
}

static void Control_EnterManual(u8 source)
{
    if (source == CONTROL_SOURCE_PS2) Control_State = CONTROL_STATE_MANUAL_PS2;
    else if (source == CONTROL_SOURCE_APP) Control_State = CONTROL_STATE_MANUAL_APP;
    else if (source == CONTROL_SOURCE_REMOTE) Control_State = CONTROL_STATE_MANUAL_REMOTE;
    else return;

    Control_Stop_Reason = CONTROL_STOP_NONE;
    Ros_Heartbeat_Started = 0;
    command_lost_count = COMMAND_LOST_TIMEOUT_TICKS;
    Auto_Last_Command_Tick = Control_Tick_100Hz;
    Manual_Input_Age = 0;
    Auto_Enable_Min_Seq = Host_Frame_Write_Seq;
    Control_ZeroMotion();
    Control_SetSourceFlags();
}

static void Control_EnterAutoWait(void)
{
    Control_State = CONTROL_STATE_AUTO_WAIT_FIRST;
    Control_Stop_Reason = CONTROL_STOP_NONE;
    Ros_Heartbeat_Started = 0;
    command_lost_count = 0;
    Auto_Last_Command_Tick = Control_Tick_100Hz;
    Control_ZeroMotion();
    Control_SetSourceFlags();
    Control_DiscardManualEvents();
    if (GamePadInterface != 0)
        GamePadInterface->StartFlag = 0;
}

void Control_SubmitHostFrameFromISR(const u8 *frame)
{
    u8 index;
    u8 slot;
    u32 sequence;

    sequence = Host_Frame_Write_Seq;
    if ((sequence - Host_Frame_Read_Seq) >= CONTROL_HOST_QUEUE_DEPTH)
    {
        Host_Frame_Overflow_Seq++;
        return;
    }
    slot = (u8)(sequence & CONTROL_HOST_QUEUE_MASK);
    for (index = 0; index < RECEIVE_DATA_SIZE; index++)
        Host_Frame_Queue[slot][index] = frame[index];
    Host_Frame_Tick[slot] = Control_Tick_100Hz;
    __DMB();
    Host_Frame_Write_Seq = sequence + 1U;
}

void Control_RequestManual(u8 source)
{
    if (Control_ValidManualSource(source))
        Manual_Request_Seq[source]++;
}

void Control_ManualHeartbeat(u8 source)
{
    if (Control_ValidManualSource(source))
        Manual_Heartbeat_Seq[source]++;
}

void Control_ManualDisconnect(u8 source)
{
    if (Control_ValidManualSource(source))
        Manual_Disconnect_Seq[source]++;
}

u8 Control_LegacyMotionAllowed(void)
{
#if CONTROL_ENABLE_LEGACY_MOTION_INPUTS
    return (Control_State == CONTROL_STATE_AUTO_READY &&
            Ros_Heartbeat_Started == 0 &&
            Host_Frame_Write_Seq == Host_Frame_Read_Seq &&
            Host_Frame_Overflow_Seq == Host_Frame_Overflow_Read);
#else
    return 0;
#endif
}

static void Control_ProcessManualEvents(u8 age_manual)
{
    u32 request_snapshot[CONTROL_MANUAL_SOURCE_COUNT];
    u32 heartbeat_snapshot[CONTROL_MANUAL_SOURCE_COUNT];
    u32 disconnect_snapshot[CONTROL_MANUAL_SOURCE_COUNT];
    u8 source;
    u8 owner;
    u8 selected;
    u8 heartbeat_pending;
    u8 disconnect_pending;

    for (source = CONTROL_SOURCE_PS2; source < CONTROL_MANUAL_SOURCE_COUNT; source++)
    {
        request_snapshot[source] = Manual_Request_Seq[source];
        heartbeat_snapshot[source] = Manual_Heartbeat_Seq[source];
        disconnect_snapshot[source] = Manual_Disconnect_Seq[source];
    }

    owner = Control_StateManualSource();
    if (owner != CONTROL_SOURCE_NONE)
    {
        heartbeat_pending = (heartbeat_snapshot[owner] != Manual_Heartbeat_Read[owner]);
        disconnect_pending = (disconnect_snapshot[owner] != Manual_Disconnect_Read[owner]);

        for (source = CONTROL_SOURCE_PS2; source < CONTROL_MANUAL_SOURCE_COUNT; source++)
        {
            Manual_Request_Read[source] = request_snapshot[source];
            Manual_Heartbeat_Read[source] = heartbeat_snapshot[source];
            Manual_Disconnect_Read[source] = disconnect_snapshot[source];
        }

        if (disconnect_pending)
        {
            Control_StopAndLatch(CONTROL_STOP_MANUAL_LOST);
            return;
        }
        if (heartbeat_pending)
            Manual_Input_Age = 0;
        else if (age_manual && Manual_Input_Age < MANUAL_INPUT_TIMEOUT_TICKS)
            Manual_Input_Age++;

        if (Manual_Input_Age >= MANUAL_INPUT_TIMEOUT_TICKS)
        {
            disconnect_pending =
                (Manual_Disconnect_Seq[owner] != Manual_Disconnect_Read[owner]);
            heartbeat_pending =
                (Manual_Heartbeat_Seq[owner] != Manual_Heartbeat_Read[owner]);
            if (disconnect_pending)
            {
                Manual_Disconnect_Read[owner] = Manual_Disconnect_Seq[owner];
                Control_StopAndLatch(CONTROL_STOP_MANUAL_LOST);
            }
            else if (heartbeat_pending)
            {
                Manual_Heartbeat_Read[owner] = Manual_Heartbeat_Seq[owner];
                Manual_Input_Age = 0;
            }
            else
            {
                Control_StopAndLatch(CONTROL_STOP_MANUAL_LOST);
            }
        }
        return;
    }

    selected = CONTROL_SOURCE_NONE;
    if (request_snapshot[CONTROL_SOURCE_PS2] != Manual_Request_Read[CONTROL_SOURCE_PS2])
        selected = CONTROL_SOURCE_PS2;
    else if (request_snapshot[CONTROL_SOURCE_APP] != Manual_Request_Read[CONTROL_SOURCE_APP])
        selected = CONTROL_SOURCE_APP;
    else if (request_snapshot[CONTROL_SOURCE_REMOTE] != Manual_Request_Read[CONTROL_SOURCE_REMOTE])
        selected = CONTROL_SOURCE_REMOTE;

    heartbeat_pending = 0;
    disconnect_pending = 0;
    if (selected != CONTROL_SOURCE_NONE)
    {
        heartbeat_pending = (heartbeat_snapshot[selected] != Manual_Heartbeat_Read[selected]);
        disconnect_pending = (disconnect_snapshot[selected] != Manual_Disconnect_Read[selected]);
    }

    for (source = CONTROL_SOURCE_PS2; source < CONTROL_MANUAL_SOURCE_COUNT; source++)
    {
        Manual_Request_Read[source] = request_snapshot[source];
        Manual_Heartbeat_Read[source] = heartbeat_snapshot[source];
        Manual_Disconnect_Read[source] = disconnect_snapshot[source];
    }

    if (selected != CONTROL_SOURCE_NONE)
    {
        if (disconnect_pending)
            Control_StopAndLatch(CONTROL_STOP_MANUAL_LOST);
        else
            Control_EnterManual(selected);
    }
}

static u8 Control_FrameHasZeroVelocity(const u8 *frame)
{
    u8 index;
    for (index = 3; index <= 8; index++)
    {
        if (frame[index] != 0)
            return 0;
    }
    return 1;
}

static void Control_ProcessHostFrame(const u8 *frame, u32 frame_sequence, u32 frame_tick)
{
    float velocity_x;
    float velocity_y;
    float velocity_z;
    u8 mode;
    u8 security;

    mode = frame[1];
    security = frame[2];

    if (mode == 0 && security == ROS_AUTO_ENABLE_SECURITY &&
        Control_FrameHasZeroVelocity(frame))
    {
        if (Control_State == CONTROL_STATE_STOP_LATCHED &&
            Control_SequenceAfter(frame_sequence, Auto_Enable_Min_Seq))
        {
            Control_EnterAutoWait();
            Auto_Last_Command_Tick = frame_tick;
        }
        return;
    }

    if (security != 0)
        return;

    if (mode == 0 &&
        (Control_State == CONTROL_STATE_AUTO_READY ||
         Control_State == CONTROL_STATE_AUTO_WAIT_FIRST))
    {
        velocity_x = XYZ_Target_Speed_transition(frame[3], frame[4]);
        velocity_y = XYZ_Target_Speed_transition(frame[5], frame[6]);
        velocity_z = XYZ_Target_Speed_transition(frame[7], frame[8]);

        Control_State = CONTROL_STATE_AUTO_READY;
        Control_Stop_Reason = CONTROL_STOP_NONE;
        Allow_Recharge = 0;
        nav_walk = 0;
        Move_X = velocity_x;
        Move_Y = velocity_y;
        if (Car_Mode == Akm_Car)
            Move_Z = Vz_to_Akm_Angle(Move_X, velocity_z);
        else
            Move_Z = velocity_z;
        command_lost_count = 0;
        Auto_Last_Command_Tick = frame_tick;
        Ros_Heartbeat_Started = 1;
        Control_SetSourceFlags();
        return;
    }

    if ((mode == 1 || mode == 2) &&
        Control_State == CONTROL_STATE_AUTO_READY)
    {
        velocity_x = XYZ_Target_Speed_transition(frame[3], frame[4]);
        velocity_y = XYZ_Target_Speed_transition(frame[5], frame[6]);
        velocity_z = XYZ_Target_Speed_transition(frame[7], frame[8]);

        Allow_Recharge = 1;
        if (mode == 1 && RED_STATE == 0)
            nav_walk = 1;
        Recharge_UP_Move_X = velocity_x;
        Recharge_UP_Move_Y = velocity_y;
        if (Car_Mode == Akm_Car)
            Recharge_UP_Move_Z = Vz_to_Akm_Angle(velocity_x, velocity_z);
        else
            Recharge_UP_Move_Z = velocity_z;
        command_lost_count = 0;
        Auto_Last_Command_Tick = frame_tick;
        Ros_Heartbeat_Started = 1;
    }

    if (mode == 3 && Control_State == CONTROL_STATE_AUTO_READY)
    {
        Red_Docker_X = XYZ_Target_Speed_transition(frame[3], frame[4]);
        Red_Docker_Y = XYZ_Target_Speed_transition(frame[5], frame[6]);
        Red_Docker_Z = XYZ_Target_Speed_transition(frame[7], frame[8]);
    }
}

static void Control_ProcessHostQueue(void)
{
    u8 frame[RECEIVE_DATA_SIZE];
    u8 index;
    u8 slot;
    u32 pending;
    u32 write_sequence;
    u32 frame_sequence;
    u32 frame_tick;
    u32 overflow_sequence;

    overflow_sequence = Host_Frame_Overflow_Seq;
    if (overflow_sequence != Host_Frame_Overflow_Read)
    {
        Host_Frame_Overflow_Read = overflow_sequence;
        Host_Frame_Read_Seq = Host_Frame_Write_Seq;
        Control_StopAndLatch(CONTROL_STOP_HOST_QUEUE_OVERFLOW);
        return;
    }
    write_sequence = Host_Frame_Write_Seq;
    pending = write_sequence - Host_Frame_Read_Seq;
    if (pending > CONTROL_HOST_QUEUE_DEPTH)
    {
        Host_Frame_Read_Seq = write_sequence;
        Control_StopAndLatch(CONTROL_STOP_HOST_QUEUE_OVERFLOW);
        return;
    }

    while (Host_Frame_Read_Seq != write_sequence)
    {
        if (Host_Frame_Overflow_Seq != overflow_sequence)
        {
            Host_Frame_Overflow_Read = Host_Frame_Overflow_Seq;
            Host_Frame_Read_Seq = Host_Frame_Write_Seq;
            Control_StopAndLatch(CONTROL_STOP_HOST_QUEUE_OVERFLOW);
            return;
        }
        slot = (u8)(Host_Frame_Read_Seq & CONTROL_HOST_QUEUE_MASK);
    __DMB();
    frame_tick = Host_Frame_Tick[slot];
        for (index = 0; index < RECEIVE_DATA_SIZE; index++)
            frame[index] = Host_Frame_Queue[slot][index];
    __DMB();
        frame_sequence = Host_Frame_Read_Seq + 1U;
        Host_Frame_Read_Seq++;
        Control_ProcessHostFrame(frame, frame_sequence, frame_tick);
    }

    if (Host_Frame_Overflow_Seq != overflow_sequence)
    {
        Host_Frame_Overflow_Read = Host_Frame_Overflow_Seq;
        Host_Frame_Read_Seq = Host_Frame_Write_Seq;
        Control_StopAndLatch(CONTROL_STOP_HOST_QUEUE_OVERFLOW);
    }
}

void Control_Arbitrate100Hz(void)
{
    u32 elapsed_ticks;

    Control_Tick_100Hz++;
    Control_ProcessManualEvents(1);
    Control_ProcessHostQueue();
    Control_ProcessManualEvents(0);
    if (Host_Frame_Write_Seq != Host_Frame_Read_Seq ||
        Host_Frame_Overflow_Seq != Host_Frame_Overflow_Read)
    {
        Control_ProcessHostQueue();
        Control_ProcessManualEvents(0);
    }
    Control_SetSourceFlags();

    if (Control_State == CONTROL_STATE_AUTO_READY && Ros_Heartbeat_Started)
    {
        elapsed_ticks = Control_Tick_100Hz - Auto_Last_Command_Tick;
        if (elapsed_ticks > COMMAND_LOST_TIMEOUT_TICKS)
            elapsed_ticks = COMMAND_LOST_TIMEOUT_TICKS;
        command_lost_count = (u16)elapsed_ticks;

        if (elapsed_ticks >= COMMAND_LOST_TIMEOUT_TICKS)
        {
            /* A final snapshot lets an already queued valid command win; an
             * invalid frame flood cannot postpone the stop any further. */
            if (Host_Frame_Write_Seq != Host_Frame_Read_Seq ||
                Host_Frame_Overflow_Seq != Host_Frame_Overflow_Read)
            {
                Control_ProcessHostQueue();
                Control_ProcessManualEvents(0);
            }
            elapsed_ticks = Control_Tick_100Hz - Auto_Last_Command_Tick;
            if (Control_State == CONTROL_STATE_AUTO_READY &&
                Ros_Heartbeat_Started &&
                elapsed_ticks >= COMMAND_LOST_TIMEOUT_TICKS)
                Control_StopAndLatch(CONTROL_STOP_AUTO_TIMEOUT);
        }
    }
    else if (Control_State == CONTROL_STATE_AUTO_WAIT_FIRST)
    {
        elapsed_ticks = Control_Tick_100Hz - Auto_Last_Command_Tick;
        if (elapsed_ticks >= COMMAND_LOST_TIMEOUT_TICKS)
            Control_StopAndLatch(CONTROL_STOP_AUTO_TIMEOUT);
        else
            Control_ZeroMotion();
    }
    else if (Control_State == CONTROL_STATE_STOP_LATCHED)
    {
        Control_ZeroMotion();
    }
}

//Bluetooth remote control associated flag bits
//蓝牙遥控相关的标志位
u8 Flag_Left, Flag_Right, Flag_Direction=0, Turn_Flag;

//Sends the parameter's flag bit to the Bluetooth APP
//向蓝牙APP发送参数的标志位
u8 PID_Send;

//The PS2 gamepad controls related variables
//PS2手柄控制相关变量
float PS2_LX,PS2_LY,PS2_RX,PS2_RY,PS2_KEY;

//Self-check the relevant flag variables
//自检相关标志变量
int Check=0;

//Check the result code
//自检结果代码
long int ErrorCode=0;

u8 Get_Charging_HardWare=0;//是否存在自动回充装备
u8 charger_check=0;//实时检测充电装备在线状态

//系统相关变量
SYS_VAL_t SysVal;

//电位器记录数值
int POT_val;

//电机满转值
int Full_rotation = 16799;

void systemInit(void)
{
	//Interrupt priority group setti  ng
	//中断优先级分组设置
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_4);

	//Delay function initialization
	//延时函数初始化
    delay_init(168);
	//IIC initialization for IMU
    //IIC初始化，用于IMU
    I2C_GPIOInit();
	//系统相关软件参数初始化
	SYS_VAL_t_Init(&SysVal);
	
    //Serial port 1 initialization, communication baud rate 115200,
    //can be used to communicate with ROS terminal
    //串口1初始化，通信波特率115200，可用于与ROS端通信
    uart1_init(115200);

    //Serial port 2 initialization, communication baud rate 9600,
    //used to communicate with Bluetooth APP terminal
    //串口2初始化，通信波特率9600，用于与蓝牙APP端通信
    uart2_init(9600);

    //Serial port 3 is initialized and the baud rate is 115200.
    //Serial port 3 is the default port used to communicate with ROS terminal
    //串口3初始化，通信波特率115200，串口3为默认用于与ROS端通信的串口
    uart3_init(115200);
//	uart3_init(230400);
	
    //Serial port 5 initialization, communication baud rate 115200,
    //can be used to communicate with ROS terminal
    //串口5初始化，通信波特率115200，可用于与ROS端通信
    uart5_init(115200);
	
	//如果IMU为MPU6050,则是旧版C30D
	if( MPU6050_DEFAULT_ADDRESS == MPU6050_getDeviceID() )
	{
		SysVal.HardWare_Ver = V1_0;
		//Initialize the hardware interface to the PS2 controller
		//初始化与PS2手柄连接的硬件接口
		PS2_Init();
		//PS2 gamepad configuration is initialized and configured in analog mode
		//PS2手柄配置初始化,配置为模拟量模式	
		PS2_SetInit();		 
		//Initialize the hardware interface connected to the LED lamp
		//初始化与LED灯连接的硬件接口
		V1_0_LED_Init(); 
		//MPU6050 is initialized to read the vehicle's three-axis attitude,
		//three-axis angular velocity and three-axis acceleration information
		//MPU6050初始化，用于读取小车三轴角速度、三轴加速度信息
		MPU6050_initialize();
	}
	//如果IMU型号为ICM20948,则是新版C30D
	else if( REG_VAL_WIA == ICM20948_getDeviceID() )//读取ICM20948 id
	{
		SysVal.HardWare_Ver = V1_1;
		
		//USBHOS初始化
		MX_USB_HOST_Init();
		
		//Initialize the hardware interface connected to the LED lamp
		//初始化与LED灯连接的硬件接口
		V1_1_LED_Init();
		//MPU6050 is initialized to read the vehicle's three-axis attitude,
		//three-axis angular velocity and three-axis acceleration information
		//ICM20948初始化，用于读取小车三轴角速度、三轴加速度信息
		invMSInit();
	}
	else //无法识别的陀螺仪,复位系统
	{
		NVIC_SystemReset();
	}         

    //Initialize the hardware interface connected to the buzzer
    //初始化与蜂鸣器连接的硬件接口
    Buzzer_Init();

    //Initialize the hardware interface connected to the enable switch
    //初始化与使能开关连接的硬件接口
    Enable_Pin();

    //Initialize the hardware interface connected to the OLED display
    //初始化与OLED显示屏连接的硬件接口
    OLED_Init();

    //Initialize the hardware interface connected to the user's key
    //初始化与用户按键连接的硬件接口
    KEY_Init();
		if(KEY==0)			Check=1;
		else if(KEY==1)	Check=0;

    //ADC pin initialization, used to read the battery voltage and potentiometer gear,
    //potentiometer gear determines the car after the boot of the car model
    //ADC引脚初始化，用于读取电池电压与电位器档位，电位器档位决定小车开机后的小车适配型号
    Adc_Init();
    Adc_POWER_Init();


	//Initialize the CAN communication interface
	//CAN通信接口初始化
	//波特率计算方式：42M/[(参数1+参数2+参数3)*参数4]
	//示例：42M/[(1+7+6)*3] = 1M （只跟前4个参数有关）
    CAN1_Mode_Init(1,7,6,3,0);

    //According to the tap position of the potentiometer, determine which type of car needs to be matched,
    //and then initialize the corresponding parameters
    //根据电位器的档位判断需要适配的是哪一种型号的小车，然后进行对应的参数初始化
    Robot_Select();

    //Encoder A is initialized to read the real time speed of motor C
    //编码器A初始化，用于读取电机C的实时速度
    Encoder_Init_TIM2();
    //Encoder B is initialized to read the real time speed of motor D
    //编码器B初始化，用于读取电机D的实时速度
    Encoder_Init_TIM3();
    //Encoder C is initialized to read the real time speed of motor B
    //编码器C初始化，用于读取电机B的实时速度
    Encoder_Init_TIM4();
    //Encoder D is initialized to read the real time speed of motor A
    //编码器D初始化，用于读取电机A的实时速度
    Encoder_Init_TIM5();

    //定时器12用作舵机的PWM接口
    TIM12_SERVO_Init(9999,84-1);  //APB1的时钟频率为84M , 频率=84M/((9999+1)*(83+1))=100Hz

    //普通小车默认定时器8用作航模接口
    // TIM8_SERVO_Init(9999,168-1);//APB2的时钟频率为168M , 频率=168M/((9999+1)*(167+1))=100Hz
    //Initialize the model remote control interface
    //初始化航模遥控接口
    TIM8_Cap_Init(9999,168-1);  //高级定时器TIM8的时钟频率为168M

    //Initialize motor speed control and, for controlling motor speed, PWM frequency 10kHz
    //初始化电机速度控制以及，用于控制电机速度，PWM频率10KHZ
    //APB2时钟频率为168M，满PWM为16799，频率=168M/((16799+1)*(0+1))=10k
    TIM1_PWM_Init(16799,0);
    TIM9_PWM_Init(16799,0);
    TIM10_PWM_Init(16799,0);
    TIM11_PWM_Init(16799,0);

    //采集1次电池电压作为电压平滑滤波的初始化参考值
    base_vol = Get_battery_volt();

    //访问是否存在自动回充装备
    delay_ms(100);
    Find_Charging_HardWare();
	
	//读取Flash数据
	FlashParam_Read();
}

void Find_Charging_HardWare(void)
{
    u8 tmpdata[8]= {1,1,1,1,1,1,1,1};
    CAN1_Send_EXTid_Num(0x12345678,tmpdata);
}

