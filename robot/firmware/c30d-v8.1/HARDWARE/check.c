#include "check.h"

//定义自检相关结构体
Check_parameter check_parameter;		
//蜂鸣器计数值
u32 Buzzer_count1 = 0;

void Check_task(void *pvParameters)
{
	u32 lastWakeTime = getSysTickCnt();
	for(int i=0; i<6; i++)
	{
		check_parameter.Servo_count[i] = 500;
	}
	check_parameter.Motor_forward = 300;
	check_parameter.Motor_retreat = 500;
	while(1)
	{
		// This task runs at a frequency of 100Hz (10ms control once)
		//此任务以100Hz的频率运行（10ms控制一次）
		vTaskDelayUntil(&lastWakeTime, F2T(RATE_100_HZ));
		//Time count is no longer needed after 30 seconds
		//时间计数，30秒后不再需要
		if(SysVal.Time_count<3000) SysVal.Time_count++;
		
		Buzzer_count1++;
		//Get the encoder data, that is, the real time wheel speed, 
		//and convert to transposition international units
		//获取编码器数据，即车轮实时速度，并转换位国际单位
		Get_Velocity_Form_Encoder();
		//Click the user button to proceed to the next step
		//Double-click to exit the self-check mode
		//单击用户按键进入下一步，双击退出自检模式
		Check_Key();
		switch(check_parameter.Process)
		{
			case 3:				//电机自检
#if CONTROL_ENABLE_MAINTENANCE_MOTOR_TEST
				if(check_parameter.Motor_forward > 0)				check_parameter.Motor_forward--,Full_rotation = 16799;		//正转
				else if(check_parameter.Motor_retreat > 0)	check_parameter.Motor_retreat--,Full_rotation = -16799;		//反转
				switch(Car_Mode)
				{
					case Mec_Car:case Mec_Car_V550: 
						Set_Pwm( Full_rotation, -Full_rotation, -Full_rotation, Full_rotation, 0    ); break; 									//Mecanum wheel car       //麦克纳姆轮小车
					case Omni_Car:      Set_Pwm(-Full_rotation,  Full_rotation, -Full_rotation, Full_rotation, 0    ); break; //Omni car                //全向轮小车
					case Akm_Car:       Set_Pwm( Full_rotation,  Full_rotation,  Full_rotation, Full_rotation, 0    ); break; //Ackermann structure car //阿克曼小车
					case Diff_Car:      Set_Pwm( Full_rotation,  Full_rotation,  Full_rotation, Full_rotation, 0    ); break; //Differential car        //两轮差速小车
					case FourWheel_Car:case FourWheel_Car_V550:
						Set_Pwm( Full_rotation, -Full_rotation, -Full_rotation, Full_rotation, 0    ); break; 									//FourWheel car           //四驱车 
					case Tank_Car:      Set_Pwm( Full_rotation,  Full_rotation,  Full_rotation, Full_rotation, 0    ); break; //Tank Car                //履带车
				} 
				if(!(check_parameter.Motor_forward > 0) && !(check_parameter.Motor_retreat > 0))				Set_Pwm(0,0,0,0,0);
				break;
#else
				Set_Pwm(0,0,0,0,0);
				break;
#endif
			case 4:				//退出电机自检，控制电机停下
				Set_Pwm(0,0,0,0,0);
				break;
			case 6:				//初始化舵机
				TIM8_SERVO_Init(9999, 168-1);
				break;
			case 7:
				//舵机正转
				if(check_parameter.Servo_Dir[0]==0 && check_parameter.Servo_count[0]<2500)
					check_parameter.Servo_count[0] = check_parameter.Servo_count[0] + 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[0]==0 && check_parameter.Servo_count[0]>=2500)
					check_parameter.Servo_Dir[0] = 1;
				//舵机反转
				if(check_parameter.Servo_Dir[0]==1 && check_parameter.Servo_count[0]>500)
					check_parameter.Servo_count[0] = check_parameter.Servo_count[0] - 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[0]==1 && check_parameter.Servo_count[0]<=500)
					check_parameter.Servo_Dir[0] = 2;
				//设置占空比控制舵机转动
				TIM12->CCR2=check_parameter.Servo_count[0];
				break;
			case 8:
				//如果上一个舵机没有转完就跳到下一个舵机的验证则给上一个舵机做复位
				if((check_parameter.Servo_Dir[0]!=2))				
					check_parameter.Servo_count[0]=500,TIM12->CCR2=check_parameter.Servo_count[0];
				//舵机正转
				if(check_parameter.Servo_Dir[1]==0 && check_parameter.Servo_count[1]<2500)
					check_parameter.Servo_count[1] = check_parameter.Servo_count[1] + 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[1]==0 && check_parameter.Servo_count[1]>=2500)
					check_parameter.Servo_Dir[1] = 1;
				//舵机反转
				if(check_parameter.Servo_Dir[1]==1 && check_parameter.Servo_count[1]>500)
					check_parameter.Servo_count[1] = check_parameter.Servo_count[1] - 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[1]==1 && check_parameter.Servo_count[1]<=500)
					check_parameter.Servo_Dir[1] = 2;
				//设置占空比控制舵机转动
				TIM12->CCR1=check_parameter.Servo_count[1];
				break;
			case 9:
				//如果上一个舵机没有转完就跳到下一个舵机的验证则给上一个舵机做复位
				if((check_parameter.Servo_Dir[1]!=2))				
					check_parameter.Servo_count[1]=500,TIM12->CCR1=check_parameter.Servo_count[1];
				//舵机正转
				if(check_parameter.Servo_Dir[2]==0 && check_parameter.Servo_count[2]<2500)
					check_parameter.Servo_count[2] = check_parameter.Servo_count[2] + 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[2]==0 && check_parameter.Servo_count[2]>=2500)
					check_parameter.Servo_Dir[2] = 1;
				//舵机反转
				if(check_parameter.Servo_Dir[2]==1 && check_parameter.Servo_count[2]>500)
					check_parameter.Servo_count[2] = check_parameter.Servo_count[2] - 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[2]==1 && check_parameter.Servo_count[2]<=500)
					check_parameter.Servo_Dir[2] = 2;
				//设置占空比控制舵机转动
				TIM8->CCR4=check_parameter.Servo_count[2];
				break;
			case 10:
				//如果上一个舵机没有转完就跳到下一个舵机的验证则给上一个舵机做复位
				if((check_parameter.Servo_Dir[2]!=2))				
					check_parameter.Servo_count[2]=500,TIM8->CCR4=check_parameter.Servo_count[2];
				//舵机正转
				if(check_parameter.Servo_Dir[3]==0 && check_parameter.Servo_count[3]<2500)
					check_parameter.Servo_count[3] = check_parameter.Servo_count[3] + 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[3]==0 && check_parameter.Servo_count[3]>=2500)
					check_parameter.Servo_Dir[3] = 1;
				//舵机反转
				if(check_parameter.Servo_Dir[3]==1 && check_parameter.Servo_count[3]>500)
					check_parameter.Servo_count[3] = check_parameter.Servo_count[3] - 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[3]==1 && check_parameter.Servo_count[3]<=500)
					check_parameter.Servo_Dir[3] = 2;
				//设置占空比控制舵机转动
				TIM8->CCR3=check_parameter.Servo_count[3];
				break;
			case 11:
				//如果上一个舵机没有转完就跳到下一个舵机的验证则给上一个舵机做复位
				if((check_parameter.Servo_Dir[3]!=2))				
					check_parameter.Servo_count[3]=500,TIM8->CCR3=check_parameter.Servo_count[3];
				//舵机正转
				if(check_parameter.Servo_Dir[4]==0 && check_parameter.Servo_count[4]<2500)
					check_parameter.Servo_count[4] = check_parameter.Servo_count[4] + 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[4]==0 && check_parameter.Servo_count[4]>=2500)
					check_parameter.Servo_Dir[4] = 1;
				//舵机反转
				if(check_parameter.Servo_Dir[4]==1 && check_parameter.Servo_count[4]>500)
					check_parameter.Servo_count[4] = check_parameter.Servo_count[4] - 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[4]==1 && check_parameter.Servo_count[4]<=500)
					check_parameter.Servo_Dir[4] = 2;
				//设置占空比控制舵机转动
				TIM8->CCR2=check_parameter.Servo_count[4];
				break;
			case 12:
				//如果上一个舵机没有转完就跳到下一个舵机的验证则给上一个舵机做复位
				if((check_parameter.Servo_Dir[4]!=2))				
					check_parameter.Servo_count[4]=500,TIM8->CCR2=check_parameter.Servo_count[4];
				//舵机正转
				if(check_parameter.Servo_Dir[5]==0 && check_parameter.Servo_count[5]<2500)
					check_parameter.Servo_count[5] = check_parameter.Servo_count[5] + 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[5]==0 && check_parameter.Servo_count[5]>=2500)
					check_parameter.Servo_Dir[5] = 1;
				//舵机反转
				if(check_parameter.Servo_Dir[5]==1 && check_parameter.Servo_count[5]>500)
					check_parameter.Servo_count[5] = check_parameter.Servo_count[5] - 5;
				//正转结束，切换标志位
				if(check_parameter.Servo_Dir[5]==1 && check_parameter.Servo_count[5]<=500)
					check_parameter.Servo_Dir[5] = 2;
				//设置占空比控制舵机转动
				TIM8->CCR1=check_parameter.Servo_count[5];
				break;
			case 13:				//舵机自检结束复位所有舵机及相关参数
				for(int i=0; i<6; i++){
					check_parameter.Servo_Dir[i] = 0;
					check_parameter.Servo_count[i] = 500;
				}
				TIM8->CCR1 = check_parameter.Servo_count[5];
				TIM8->CCR2 = check_parameter.Servo_count[4];
				TIM8->CCR3 = check_parameter.Servo_count[3];
				TIM8->CCR4 = check_parameter.Servo_count[2];
				TIM12->CCR1 = check_parameter.Servo_count[1];
				TIM12->CCR2 = check_parameter.Servo_count[0];
				break;;
			case 14:				//蜂鸣器自检相关
				if((Buzzer_count1/100)%2)			Buzzer = 1;
				else													Buzzer = 0;
				break;
			case 15:				//退出蜂鸣器自检时关闭蜂鸣器
				Buzzer = 0;
				break;
			case 19:				//串口3自检，检测到标志位则缓存区字符
				if(check_parameter.Usart3_check_flag==1)
				{
					USART3_Return();
					check_parameter.Usart3_check_flag = 0;
					check_parameter.Usart3_rec_count = 0;
				}
				break;
			default:
				break;
		}
	}
}

void Check_Key(void)
{
	u8 tmp;

  //传入任务的频率
  tmp=KEY_Scan(RATE_100_HZ,0);
	if(tmp==single_click)				//单击
	{
		check_parameter.Process++;
		if(check_parameter.Process==21)			//自检结束并退出
		{
			delay_ms(50);
			NVIC_SystemReset();
		}
	}
	else if(tmp==double_click)
	{
		delay_ms(50);
		NVIC_SystemReset();
	}
}




