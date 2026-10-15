#include "WiredPS2_gamepad.h"
#include "FreeRTOS.h"
#include "task.h"

#include "system.h"

__weak void Wired_USB_PS2GamePad_KeyEvent_Callback(uint8_t keyid,GamePadKeyEventType_t event)
{
	printf("%d,%d\r\n",keyid,event);
}

static GamePadKeyStateType_t WiredPS2_GetKeyState(uint8_t bit);

//有线ps2游戏手柄
GamePadType_t Wired_USB_PS2Gamepad = {
	.LX = 127,
	.LY = 127,
	.RX = 127,
	.RY = 127,
	.LT = 0,
	.RT = 0,
	.StartFlag = 0,
	.Vib_EN = 0,
	.SetVibration = NULL,
	.getKeyState = WiredPS2_GetKeyState
};

//按键值
static uint16_t GamePad_KeyOriginalVal = 0;

//获取手柄键值状态函数
static GamePadKeyStateType_t WiredPS2_GetKeyState(uint8_t bit)
{
	if( (GamePad_KeyOriginalVal>>bit)&0x01 )
		return GamePadKeyState_Press;
	else
		return GamePadKeyState_Release;
}


//定义按键检测事件的各个按键值
static GamePad_CheckEventType_t GamePadKeyCheckEvent[16] = { 0 };


//标志位设置函数,用于辅助ps2手柄解码
static void ps2_set_bit(uint16_t* state,uint8_t state_bit,uint8_t bit)
{
	if(state_bit==1) //指定的位(bit)设置为1,其他位不变
	{
		*state |= (1U<<bit);
	}
	else //指定的位(bit)设置为0,其他位不变
	{
		*state &= ~(1U<<bit);
	}
}

//数据解码
void Wired_USB_PS2gamepad_Decode(USBH_HandleTypeDef *phost,uint8_t* buffer,uint8_t datalen)
{
	(void)phost;
	
	uint8_t tmp_bool = 0 ;
	
//	static uint8_t times = 0;
//	if( times<2 )
//	{
//		times++;
//		uint8_t vibration_data[2] = {1,1};
//		USBH_InterruptSendData(phost, vibration_data, 2, HID_Handle->OutPipe);
//	}
	
	Wired_USB_PS2Gamepad.LX = buffer[3];
	Wired_USB_PS2Gamepad.LY = 255-buffer[4];
	Wired_USB_PS2Gamepad.RX = buffer[1];
	Wired_USB_PS2Gamepad.RY = buffer[2];
	
	tmp_bool = (buffer[6]>>4)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,0); //seltec key 选择按键
	
	tmp_bool = (buffer[6]>>6)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,1); //左摇杆按键
	
	tmp_bool = (buffer[6]>>7)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,2); //右摇杆按键
	
	tmp_bool = (buffer[6]>>5)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,3); //start
	
	tmp_bool = buffer[5]&0x0F;//取出低4位
	if(tmp_bool==0x0F)//没有任何按键按下
	{
		ps2_set_bit(&GamePad_KeyOriginalVal,0,4); //↑
		ps2_set_bit(&GamePad_KeyOriginalVal,0,5); //→
		ps2_set_bit(&GamePad_KeyOriginalVal,0,6); //↓
		ps2_set_bit(&GamePad_KeyOriginalVal,0,7); //←
	}
	else if( (tmp_bool&0x01)==0 )
	{	
		switch ((tmp_bool>>1)&0x03)
		{
			case 0x00://↑
				ps2_set_bit(&GamePad_KeyOriginalVal,1,4); //↑
				break;
			case 0x01://→
				ps2_set_bit(&GamePad_KeyOriginalVal,1,5); //→
				break;
			case 0x02://↓
				ps2_set_bit(&GamePad_KeyOriginalVal,1,6); //↓
				break;
			case 0x03://←
				ps2_set_bit(&GamePad_KeyOriginalVal,1,7); //←
				break;
			default:
				break;
		}
	}
	else if( (tmp_bool&0x01)==1 ) //首位为1,代表存在左盘2个按键按下的情况
	{
		switch ((tmp_bool>>1)&0x03)
		{
			case 0x00://↑→
				ps2_set_bit(&GamePad_KeyOriginalVal,1,4);//↑
				ps2_set_bit(&GamePad_KeyOriginalVal,1,5); //→
				break;
			case 0x01://↓→
				ps2_set_bit(&GamePad_KeyOriginalVal,1,6); //↓
				ps2_set_bit(&GamePad_KeyOriginalVal,1,5); //→
				break;
			case 0x02://↓←
				ps2_set_bit(&GamePad_KeyOriginalVal,1,6); //↓
				ps2_set_bit(&GamePad_KeyOriginalVal,1,7); //←
				break;
			case 0x03://↑←
				ps2_set_bit(&GamePad_KeyOriginalVal,1,4); //↑
				ps2_set_bit(&GamePad_KeyOriginalVal,1,7); //←
				break;
			default:
				break;
		}
	}
	
	tmp_bool = (buffer[6]>>2)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,8); //左扳机2号
	if( tmp_bool ) Wired_USB_PS2Gamepad.LT = 255;
	else Wired_USB_PS2Gamepad.LT = 0;
	
	tmp_bool = (buffer[6]>>3)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,9); //右扳机2号
	if( tmp_bool ) Wired_USB_PS2Gamepad.RT = 255;
	else Wired_USB_PS2Gamepad.RT = 0;
	
	tmp_bool = (buffer[6]>>0)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,10); //左扳机1号
	
	tmp_bool = (buffer[6]>>1)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,11); //右扳机1号

	tmp_bool = (buffer[5]>>4)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,12); //一号,绿色GREEN
	
	tmp_bool = (buffer[5]>>5)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,13); //二号,红色RED

	tmp_bool = (buffer[5]>>6)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,14); //三号,蓝牙BLUE
	
	tmp_bool = (buffer[5]>>7)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,15); //四号,粉色PINK
	
	//按键回调函数触发
	for (uint8_t key = PS2KEY_SELECT; key <= PS2KEY_4PINK; key++) 
	{
		GamePadKeyEventType_t event = GamePadKey_CheckEvent(GamePad_KeyOriginalVal,
                                		&GamePadKeyCheckEvent[key], key);
		
		if (event == GamePadKeyEvent_NONE) continue;
		
		//触发回调函数
		Wired_USB_PS2GamePad_KeyEvent_Callback(key,event);
	}

}

//数据解码
void Wired_USB_V2_PS2gamepad_Decode(USBH_HandleTypeDef *phost,uint8_t* buffer,uint8_t datalen)
{
	(void)phost;
	
	uint8_t tmp_bool = 0 ;
	
//	static uint8_t times = 0;
//	if( times<2 )
//	{
//		times++;
//		uint8_t vibration_data[2] = {1,1};
//		USBH_InterruptSendData(phost, vibration_data, 2, HID_Handle->OutPipe);
//	}
	
	Wired_USB_PS2Gamepad.LX = buffer[0];
	Wired_USB_PS2Gamepad.LY = 255-buffer[1];
	Wired_USB_PS2Gamepad.RX = buffer[3];
	Wired_USB_PS2Gamepad.RY = buffer[4];
	
	tmp_bool = (buffer[6]>>4)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,0); //seltec key 选择按键
	
	tmp_bool = (buffer[6]>>6)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,1); //左摇杆按键
	
	tmp_bool = (buffer[6]>>7)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,2); //右摇杆按键
	
	tmp_bool = (buffer[6]>>5)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,3); //start
	
	tmp_bool = buffer[5]&0x0F;//取出低4位
	if(tmp_bool==0x0F)//没有任何按键按下
	{
		ps2_set_bit(&GamePad_KeyOriginalVal,0,4); //↑
		ps2_set_bit(&GamePad_KeyOriginalVal,0,5); //→
		ps2_set_bit(&GamePad_KeyOriginalVal,0,6); //↓
		ps2_set_bit(&GamePad_KeyOriginalVal,0,7); //←
	}
	else if( (tmp_bool&0x01)==0 )
	{	
		switch ((tmp_bool>>1)&0x03)
		{
			case 0x00://↑
				ps2_set_bit(&GamePad_KeyOriginalVal,1,4); //↑
				break;
			case 0x01://→
				ps2_set_bit(&GamePad_KeyOriginalVal,1,5); //→
				break;
			case 0x02://↓
				ps2_set_bit(&GamePad_KeyOriginalVal,1,6); //↓
				break;
			case 0x03://←
				ps2_set_bit(&GamePad_KeyOriginalVal,1,7); //←
				break;
			default:
				break;
		}
	}
	else if( (tmp_bool&0x01)==1 ) //首位为1,代表存在左盘2个按键按下的情况
	{
		switch ((tmp_bool>>1)&0x03)
		{
			case 0x00://↑→
				ps2_set_bit(&GamePad_KeyOriginalVal,1,4);//↑
				ps2_set_bit(&GamePad_KeyOriginalVal,1,5); //→
				break;
			case 0x01://↓→
				ps2_set_bit(&GamePad_KeyOriginalVal,1,6); //↓
				ps2_set_bit(&GamePad_KeyOriginalVal,1,5); //→
				break;
			case 0x02://↓←
				ps2_set_bit(&GamePad_KeyOriginalVal,1,6); //↓
				ps2_set_bit(&GamePad_KeyOriginalVal,1,7); //←
				break;
			case 0x03://↑←
				ps2_set_bit(&GamePad_KeyOriginalVal,1,4); //↑
				ps2_set_bit(&GamePad_KeyOriginalVal,1,7); //←
				break;
			default:
				break;
		}
	}
	
	tmp_bool = (buffer[6]>>2)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,8); //左扳机2号
	if( tmp_bool ) Wired_USB_PS2Gamepad.LT = 255;
	else Wired_USB_PS2Gamepad.LT = 0;
	
	tmp_bool = (buffer[6]>>3)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,9); //右扳机2号
	if( tmp_bool ) Wired_USB_PS2Gamepad.RT = 255;
	else Wired_USB_PS2Gamepad.RT = 0;
	
	tmp_bool = (buffer[6]>>0)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,10); //左扳机1号
	
	tmp_bool = (buffer[6]>>1)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,11); //右扳机1号

	tmp_bool = (buffer[5]>>4)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,12); //一号,绿色GREEN
	
	tmp_bool = (buffer[5]>>5)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,13); //二号,红色RED

	tmp_bool = (buffer[5]>>6)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,14); //三号,蓝牙BLUE
	
	tmp_bool = (buffer[5]>>7)&0x01;
	ps2_set_bit(&GamePad_KeyOriginalVal,tmp_bool,15); //四号,粉色PINK
	
	//按键回调函数触发
	for (uint8_t key = PS2KEY_SELECT; key <= PS2KEY_4PINK; key++) 
	{
		GamePadKeyEventType_t event = GamePadKey_CheckEvent(GamePad_KeyOriginalVal,
                                		&GamePadKeyCheckEvent[key], key);
		
		if (event == GamePadKeyEvent_NONE) continue;
		
		//触发回调函数
		Wired_USB_PS2GamePad_KeyEvent_Callback(key,event);
	}

}





