# Generated migration for Instance-based Face system

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0002_face_embeddings'),
    ]

    operations = [
        # First, remove unique_together before removing fields
        migrations.AlterUniqueTogether(
            name='face',
            unique_together=set(),
        ),

        # Remove old indexes
        migrations.AlterIndexTogether(
            name='face',
            index_together=set(),
        ),

        # Remove old fields
        migrations.RemoveField(
            model_name='face',
            name='face_index',
        ),
        migrations.RemoveField(
            model_name='face',
            name='embeddings',
        ),
        migrations.RemoveField(
            model_name='face',
            name='appearance_count',
        ),
        migrations.RemoveField(
            model_name='face',
            name='first_frame',
        ),
        migrations.RemoveField(
            model_name='face',
            name='last_frame',
        ),

        # Add new fields
        migrations.AddField(
            model_name='face',
            name='instance_id',
            field=models.IntegerField(
                default=0,
                verbose_name='Instance ID',
                help_text='Re-ID 시스템이 할당한 고유 instance 번호 (같은 사람은 항상 같은 ID)',
                db_index=True
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='face',
            name='track_ids',
            field=models.JSONField(
                default=list,
                verbose_name='Track ID 목록',
                help_text='이 instance에 속한 모든 BoTSORT track ID (JSON 배열)'
            ),
        ),
        migrations.AddField(
            model_name='face',
            name='quality_score',
            field=models.FloatField(
                default=0.0,
                verbose_name='품질 점수',
                help_text='Laplacian variance 선명도 점수 (높을수록 선명)',
                db_index=True
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='face',
            name='frame_index',
            field=models.IntegerField(
                default=0,
                verbose_name='썸네일 프레임 번호',
                help_text='최고 품질 썸네일이 추출된 프레임 번호'
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='face',
            name='bbox',
            field=models.JSONField(
                default=list,
                verbose_name='Bounding Box',
                help_text='얼굴 위치 좌표 [x1, y1, x2, y2]'
            ),
        ),
        migrations.AddField(
            model_name='face',
            name='total_frames',
            field=models.IntegerField(
                default=0,
                verbose_name='등장 프레임 수',
                help_text='영상 내 이 instance가 등장한 총 프레임 수'
            ),
        ),

        # Update unique_together with new fields
        migrations.AlterUniqueTogether(
            name='face',
            unique_together={('video', 'instance_id')},
        ),

        # Update model options
        migrations.AlterModelOptions(
            name='face',
            options={
                'verbose_name': '얼굴 Instance',
                'verbose_name_plural': '얼굴 Instance 목록',
                'ordering': ['instance_id']
            },
        ),

        # Add new indexes
        migrations.AddIndex(
            model_name='face',
            index=models.Index(fields=['video', 'instance_id'], name='faces_video_instance_idx'),
        ),
        migrations.AddIndex(
            model_name='face',
            index=models.Index(fields=['quality_score'], name='faces_quality_idx'),
        ),
    ]
