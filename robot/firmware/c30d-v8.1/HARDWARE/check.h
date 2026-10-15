#ifndef __CHECK_H
#define __CHECK_H

#include "sys.h"
#include "system.h"

typedef struct
{
	unsigned char Process;				//标志自检模式的进度
	int Motor_forward;						//电机正转计数值
	int Motor_retreat;						//电机反转计数值
	unsigned char Servo_Dir[6];		//六个舵机的方向
	int Servo_count[6];						//六个舵机的pwm值
	unsigned char Usart2_check_dir;			//蓝牙接收遥控方向标志位
	unsigned char Usart3_check_flag;		//串口3自检标志位
	unsigned char Usart3_rec_buffer[50];//串口3接收缓存区
	unsigned char Usart3_rec_count;
}Check_parameter;

#define CHECK_TASK_PRIO		4     //Task priority //任务优先级
#define CHECK_STK_SIZE 		512   //Task stack size //任务堆栈大小

extern Check_parameter check_parameter;

void Check_task(void *pvParameters);
void Check_Key(void);

#endif
