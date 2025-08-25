// Mission.cs
using UnityEngine;

public class Mission
{
    public int order_item_id;
    public string type; // INBOUND or OUTBOUND
    public int order_id;
    public string item_id;
    public int requested_quantity;
    public int processed_quantity;
    public int start_location_id;
    public int goal_location_id;
    public int start_y, start_x, goal_y, goal_x;
    public string rack_id;
    public string status; // WAIT, PENDING, READY, DOING, DONE
    public int priority;
    public int assigned_agent;
    public int requested_time;

    // 생성자
    public Mission(string[] fields)
    {
        // 필드 순서는 CSV의 컬럼 순서와 동일하게 맞춘다!
        order_item_id = int.Parse(fields[0]);
        type = fields[1];
        order_id = int.Parse(fields[2]);
        item_id = fields[3];
        requested_quantity = int.Parse(fields[4]);
        processed_quantity = int.Parse(fields[5]);
        start_location_id = int.Parse(fields[6]);
        goal_location_id = int.Parse(fields[7]);
        start_y = int.Parse(fields[8]);
        start_x = int.Parse(fields[9]);
        goal_y = int.Parse(fields[10]);
        goal_x = int.Parse(fields[11]);
        rack_id = fields[12];
        status = fields[13];
        priority = int.Parse(fields[14]);
        // assigned_agent, requested_time 컬럼이 없으면 체크 후 주석
        if (fields.Length > 15 && !string.IsNullOrWhiteSpace(fields[15]))
            assigned_agent = int.Parse(fields[15]);
        if (fields.Length > 16 && !string.IsNullOrWhiteSpace(fields[16]))
            requested_time = int.Parse(fields[16]);
    }

    // 유니티 좌표 변환: (start_x, start_y), (goal_x, goal_y)로 world position 만들기 (cell size 예시)
    public Vector3 GetStartWorldPos(float cellSize)
    {
        return new Vector3(start_x * cellSize, 0, start_y * cellSize);
    }
    public Vector3 GetGoalWorldPos(float cellSize)
    {
        return new Vector3(goal_x * cellSize, 0, goal_y * cellSize);
    }
}