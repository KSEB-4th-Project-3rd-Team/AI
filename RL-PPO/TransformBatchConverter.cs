using UnityEngine;
using UnityEditor;

public class TransformBatchConverter : EditorWindow
{
    float scaleFactor = 0.1f;   // 여기서 tileSize=0.1이면 0.1로!

    [MenuItem("Tools/Batch Transform Scale/Position")]
    static void Init()
    {
        TransformBatchConverter window = (TransformBatchConverter)EditorWindow.GetWindow(typeof(TransformBatchConverter));
        window.Show();
    }

    void OnGUI()
    {
        GUILayout.Label("Batch Transform Converter", EditorStyles.boldLabel);
        scaleFactor = EditorGUILayout.FloatField("Scale Factor", scaleFactor);

        if (GUILayout.Button("Convert Selected GameObject & Children"))
        {
            if (Selection.activeGameObject != null)
            {
                Transform root = Selection.activeGameObject.transform;
                Undo.RegisterFullObjectHierarchyUndo(root.gameObject, "Batch Transform Convert");
                ConvertAll(root, scaleFactor);
                Debug.Log("Batch Transform done!");
            }
            else
            {
                Debug.LogWarning("먼저 Hierarchy에서 변환할 부모 GameObject를 선택해줘!");
            }
        }
    }

    void ConvertAll(Transform t, float factor)
    {
        // Scale & Position 변환
        t.localPosition = t.localPosition * factor;
        t.localScale = t.localScale * factor;
        foreach (Transform child in t)
        {
            ConvertAll(child, factor);
        }
    }
}
